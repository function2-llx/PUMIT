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

from .post.array import (
    GenerateDistanceMap,
    GenerateInstanceBorder,
    GenerateInstanceCentroid,
    GenerateInstanceContour,
    GenerateInstanceType,
    GenerateSuccinctContour,
    GenerateWatershedMarkers,
    GenerateWatershedMask,
    HoVerNetInstanceMapPostProcessing,
    HoVerNetNuclearTypePostProcessing,
    Watershed,
)
from .post.dictionary import (
    GenerateDistanceMapD,
    GenerateDistanceMapd,
    GenerateDistanceMapDict,
    GenerateInstanceBorderD,
    GenerateInstanceBorderd,
    GenerateInstanceBorderDict,
    GenerateInstanceCentroidD,
    GenerateInstanceCentroidd,
    GenerateInstanceCentroidDict,
    GenerateInstanceContourD,
    GenerateInstanceContourd,
    GenerateInstanceContourDict,
    GenerateInstanceTypeD,
    GenerateInstanceTyped,
    GenerateInstanceTypeDict,
    GenerateSuccinctContourD,
    GenerateSuccinctContourd,
    GenerateSuccinctContourDict,
    GenerateWatershedMarkersD,
    GenerateWatershedMarkersd,
    GenerateWatershedMarkersDict,
    GenerateWatershedMaskD,
    GenerateWatershedMaskd,
    GenerateWatershedMaskDict,
    HoVerNetInstanceMapPostProcessingD,
    HoVerNetInstanceMapPostProcessingd,
    HoVerNetInstanceMapPostProcessingDict,
    HoVerNetNuclearTypePostProcessingD,
    HoVerNetNuclearTypePostProcessingd,
    HoVerNetNuclearTypePostProcessingDict,
    WatershedD,
    Watershedd,
    WatershedDict,
)
from .stain.array import ExtractHEStains, NormalizeHEStains
from .stain.dictionary import (
    ExtractHEStainsd,
    ExtractHEStainsD,
    ExtractHEStainsDict,
    NormalizeHEStainsd,
    NormalizeHEStainsD,
    NormalizeHEStainsDict,
)
