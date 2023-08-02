# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

from monai.auto3dseg.analyzer import (
    Analyzer,
    FgImageStats,
    FgImageStatsSumm,
    FilenameStats,
    ImageHistogram,
    ImageHistogramSumm,
    ImageStats,
    ImageStatsSumm,
    LabelStats,
    LabelStatsSumm,
)
from monai.transforms import Compose
from monai.utils.enums import DataStatsKeys

__all__ = ["SegSummarizer"]


class SegSummarizer(Compose):
    """
    SegSummarizer serializes the operations for data analysis in Auto3Dseg pipeline. It loads
    two types of analyzer functions and execute differently. The first type of analyzer is
    CaseAnalyzer which is similar to traditional monai transforms. It can be composed with other
    transforms to process the data dict which has image/label keys. The second type of analyzer
    is SummaryAnalyzer which works only on a list of dictionary. Each dictionary is the output
    of the case analyzers on a single dataset.

    Args:
        image_key: a string that user specify for the image. The DataAnalyzer will look it up in the
            datalist to locate the image files of the dataset.
        label_key: a string that user specify for the label. The DataAnalyzer will look it up in the
            datalist to locate the label files of the dataset. If label_key is None, the DataAnalyzer
            will skip looking for labels and all label-related operations.
        do_ccp: apply the connected component algorithm to process the labels/images.
        hist_bins: list of positive integers (one for each channel) for setting the number of bins used to
            compute the histogram. Defaults to [100].
        hist_range: list of lists of two floats (one for each channel) setting the intensity range to
            compute the histogram. Defaults to [-500, 500].
        histogram_only: whether to only compute histograms. Defaults to False.

    Examples:
        .. code-block:: python

            # imports

            summarizer = SegSummarizer("image", "label")
            transform_list = [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
                ToDeviced(keys=keys, device=device, non_blocking=True),
                Orientationd(keys=keys, axcodes="RAS"),
                EnsureTyped(keys=keys, data_type="tensor"),
                Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
                SqueezeDimd(keys=["label"], dim=0),
                summarizer,
            ]
            ...
            # skip some steps to set up data loader
            dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
            transform = Compose(transform_list)
            stats = []
            for batch_data in dataset:
                d = transform(batch_data[0])
                stats.append(d)
            report = summarizer.summarize(stats)
    """

    def __init__(
        self,
        image_key: str,
        label_key: str | None,
        average: bool = True,
        do_ccp: bool = True,
        hist_bins: list[int] | int | None = None,
        hist_range: list | None = None,
        histogram_only: bool = False,
    ) -> None:
        self.image_key = image_key
        self.label_key = label_key
        # set defaults
        self.hist_bins: list[int] | int = [100] if hist_bins is None else hist_bins
        self.hist_range: list = [-500, 500] if hist_range is None else hist_range
        self.histogram_only = histogram_only

        self.summary_analyzers: list[Any] = []
        super().__init__()

        if not self.histogram_only:
            self.add_analyzer(FilenameStats(image_key, DataStatsKeys.BY_CASE_IMAGE_PATH), None)
            self.add_analyzer(FilenameStats(label_key, DataStatsKeys.BY_CASE_LABEL_PATH), None)
            self.add_analyzer(ImageStats(image_key), ImageStatsSumm(average=average))

            if label_key is None:
                return

            self.add_analyzer(FgImageStats(image_key, label_key), FgImageStatsSumm(average=average))

            self.add_analyzer(
                LabelStats(image_key, label_key, do_ccp=do_ccp), LabelStatsSumm(average=average, do_ccp=do_ccp)
            )

        # compute histograms
        if self.hist_bins != 0:
            self.add_analyzer(
                ImageHistogram(image_key=image_key, hist_bins=hist_bins, hist_range=hist_range), ImageHistogramSumm()
            )

    def add_analyzer(self, case_analyzer: Analyzer, summary_analyzer: Analyzer | None) -> None:
        """
        Add new analyzers to the engine so that the callable and summarize functions will
        utilize the new analyzers for stats computations.

        Args:
            case_analyzer: analyzer that works on each data.
            summary_analyzer: analyzer that works on list of stats dict (output from case_analyzers).

        Examples:

            .. code-block:: python

                from monai.auto3dseg import Analyzer
                from monai.auto3dseg.utils import concat_val_to_np
                from monai.auto3dseg.analyzer_engine import SegSummarizer

                class UserAnalyzer(Analyzer):
                    def __init__(self, image_key="image", stats_name="user_stats"):
                        self.image_key = image_key
                        report_format = {"ndims": None}
                        super().__init__(stats_name, report_format)

                    def __call__(self, data):
                        d = dict(data)
                        report = deepcopy(self.get_report_format())
                        report["ndims"] = d[self.image_key].ndim
                        d[self.stats_name] = report
                        return d

                class UserSummaryAnalyzer(Analyzer):
                    def __init__(stats_name="user_stats"):
                        report_format = {"ndims": None}
                        super().__init__(stats_name, report_format)
                        self.update_ops("ndims", SampleOperations())

                    def __call__(self, data):
                        report = deepcopy(self.get_report_format())
                        v_np = concat_val_to_np(data, [self.stats_name, "ndims"])
                        report["ndims"] = self.ops["ndims"].evaluate(v_np)
                        return report

                summarizer = SegSummarizer()
                summarizer.add_analyzer(UserAnalyzer, UserSummaryAnalyzer)

        """
        self.transforms += (case_analyzer,)
        if summary_analyzer is not None:
            self.summary_analyzers.append(summary_analyzer)

    def summarize(self, data: list[dict]) -> dict[str, dict]:
        """
        Summarize the input list of data and generates a report ready for json/yaml export.

        Args:
            data: a list of data dicts.

        Returns:
            a dict that summarizes the stats across data samples.

        Examples:
            stats_summary:
                image_foreground_stats:
                    intensity: {...}
                image_stats:
                    channels: {...}
                    cropped_shape: {...}
                    ...
                label_stats:
                    image_intensity: {...}
                    label:
                    - image_intensity: {...}
                    - image_intensity: {...}
                    - image_intensity: {...}
                    - image_intensity: {...}
        """
        if not isinstance(data, list):
            raise ValueError(f"{self.__class__} summarize function needs input to be a list of dict")

        report: dict[str, dict] = {}
        if len(data) == 0:
            return report

        if not isinstance(data[0], dict):
            raise ValueError(f"{self.__class__} summarize function needs a list of dict. Now we have {type(data[0])}")

        for analyzer in self.summary_analyzers:
            if callable(analyzer):
                report.update({analyzer.stats_name: analyzer(data)})

        return report
