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

import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import copy
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Sequence

from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.bundle.properties import InferProperties, TrainProperties
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS, EXPR_KEY, ID_REF_KEY, ID_SEP_KEY
from monai.utils import BundleProperty, BundlePropertyConfig

__all__ = ["BundleWorkflow", "ConfigWorkflow"]

logger = get_logger(module_name=__name__)


class BundleWorkflow(ABC):
    """
    Base class for the workflow specification in bundle, it can be a training, evaluation or inference workflow.
    It defines the basic interfaces for the bundle workflow behavior: `initialize`, `run`, `finalize`, etc.
    And also provides the interface to get / set public properties to interact with a bundle workflow.

    Args:
        workflow: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for common workflow.

    """

    supported_train_type: tuple = ("train", "training")
    supported_infer_type: tuple = ("infer", "inference", "eval", "evaluation")

    def __init__(self, workflow: str | None = None):
        if workflow is None:
            self.properties = None
            self.workflow = None
            return
        if workflow.lower() in self.supported_train_type:
            self.properties = copy(TrainProperties)
            self.workflow = "train"
        elif workflow.lower() in self.supported_infer_type:
            self.properties = copy(InferProperties)
            self.workflow = "infer"
        else:
            raise ValueError(f"Unsupported workflow type: '{workflow}'.")

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize the bundle workflow before running.

        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the bundle workflow, it can be a training, evaluation or inference.

        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Finalize step after the running of bundle workflow.

        """
        raise NotImplementedError()

    @abstractmethod
    def _get_property(self, name: str, property: dict) -> Any:
        """
        With specified property name and information, get the expected property value.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        raise NotImplementedError()

    @abstractmethod
    def _set_property(self, name: str, property: dict, value: Any) -> Any:
        """
        With specified property name and information, set value for the expected property.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
            value: value to set for the property.

        """
        raise NotImplementedError()

    def __getattr__(self, name):
        if self.properties is not None and name in self.properties:
            return self._get_property(name=name, property=self.properties[name])
        else:
            return self.__getattribute__(name)  # getting regular attribute

    def __setattr__(self, name, value):
        if name != "properties" and self.properties is not None and name in self.properties:
            self._set_property(name=name, property=self.properties[name], value=value)
        else:
            super().__setattr__(name, value)  # setting regular attribute

    def get_workflow_type(self):
        """
        Get the workflow type, it can be `None`, "train", or "infer".

        """
        return self.workflow

    def add_property(self, name: str, required: str, desc: str | None = None) -> None:
        """
        Besides the default predefined properties, some 3rd party applications may need the bundle
        definition to provide additional properties for the specific use cases, if the bundle can't
        provide the property, means it can't work with the application.
        This utility adds the property for the application requirements check and access.

        Args:
            name: the name of target property.
            required: whether the property is "must-have".
            desc: descriptions for the property.
        """
        if self.properties is None:
            self.properties = {}
        if name in self.properties:
            warnings.warn(f"property '{name}' already exists in the properties list, overriding it.")
        self.properties[name] = {BundleProperty.DESC: desc, BundleProperty.REQUIRED: required}

    def check_properties(self) -> list[str] | None:
        """
        Check whether the required properties are existing in the bundle workflow.
        If no workflow type specified, return None, otherwise, return a list of required but missing properties.

        """
        if self.properties is None:
            return None
        return [n for n, p in self.properties.items() if p.get(BundleProperty.REQUIRED, False) and not hasattr(self, n)]


class ConfigWorkflow(BundleWorkflow):
    """
    Specification for the config-based bundle workflow.
    Standardized the `initialize`, `run`, `finalize` behavior in a config-based training, evaluation, or inference.
    For more information: https://docs.monai.io/en/latest/mb_specification.html.

    Args:
        run_id: ID name of the expected config expression to run, default to "run".
            to run the config, the target config must contain this ID.
        init_id: ID name of the expected config expression to initialize before running, default to "initialize".
            allow a config to have no `initialize` logic and the ID.
        final_id: ID name of the expected config expression to finalize after running, default to "finalize".
            allow a config to have no `finalize` logic and the ID.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
            Default to "configs/metadata.json", which is commonly used for bundles in MONAI model zoo.
        config_file: filepath of the config file, if it is a list of file paths, the content of them will be merged.
        logging_file: config file for `logging` module in the program. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.
            Default to "configs/logging.conf", which is commonly used for bundles in MONAI model zoo.
        tracking: if not None, enable the experiment tracking at runtime with optionally configurable and extensible.
            if "mlflow", will add `MLFlowHandler` to the parsed bundle with default tracking settings,
            if other string, treat it as file path to load the tracking settings.
            if `dict`, treat it as tracking settings.
            will patch the target config content with `tracking handlers` and the top-level items of `configs`.
            for detailed usage examples, please check the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/bundle_integrate_mlflow.ipynb.
        workflow: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for common workflow.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``, ``--net %/data/other.json#net_arg``

    """

    def __init__(
        self,
        config_file: str | Sequence[str],
        meta_file: str | Sequence[str] | None = "configs/metadata.json",
        logging_file: str | None = "configs/logging.conf",
        init_id: str = "initialize",
        run_id: str = "run",
        final_id: str = "finalize",
        tracking: str | dict | None = None,
        workflow: str | None = None,
        **override: Any,
    ) -> None:
        super().__init__(workflow=workflow)
        if logging_file is not None:
            if not os.path.exists(logging_file):
                if logging_file == "configs/logging.conf":
                    warnings.warn("Default logging file in 'configs/logging.conf' does not exist, skipping logging.")
                else:
                    raise FileNotFoundError(f"Cannot find the logging config file: {logging_file}.")
            else:
                logger.info(f"Setting logging properties based on config: {logging_file}.")
                fileConfig(logging_file, disable_existing_loggers=False)

        self.parser = ConfigParser()
        self.parser.read_config(f=config_file)
        if meta_file is not None:
            if isinstance(meta_file, str) and not os.path.exists(meta_file):
                if meta_file == "configs/metadata.json":
                    warnings.warn("Default metadata file in 'configs/metadata.json' does not exist, skipping loading.")
                else:
                    raise FileNotFoundError(f"Cannot find the metadata config file: {meta_file}.")
            else:
                self.parser.read_meta(f=meta_file)

        # the rest key-values in the _args are to override config content
        self.parser.update(pairs=override)
        self.init_id = init_id
        self.run_id = run_id
        self.final_id = final_id
        # set tracking configs for experiment management
        if tracking is not None:
            if isinstance(tracking, str) and tracking in DEFAULT_EXP_MGMT_SETTINGS:
                settings_ = DEFAULT_EXP_MGMT_SETTINGS[tracking]
            else:
                settings_ = ConfigParser.load_config_files(tracking)
            self.patch_bundle_tracking(parser=self.parser, settings=settings_)
        self._is_initialized: bool = False

    def initialize(self) -> Any:
        """
        Initialize the bundle workflow before running.

        """
        # reset the "reference_resolver" buffer at initialization stage
        self.parser.parse(reset=True)
        self._is_initialized = True
        return self._run_expr(id=self.init_id)

    def run(self) -> Any:
        """
        Run the bundle workflow, it can be a training, evaluation or inference.

        """
        if self.run_id not in self.parser:
            raise ValueError(f"run ID '{self.run_id}' doesn't exist in the config file.")
        return self._run_expr(id=self.run_id)

    def finalize(self) -> Any:
        """
        Finalize step after the running of bundle workflow.

        """
        return self._run_expr(id=self.final_id)

    def check_properties(self) -> list[str] | None:
        """
        Check whether the required properties are existing in the bundle workflow.
        If the optional properties have reference in the config, will also check whether the properties are existing.
        If no workflow type specified, return None, otherwise, return a list of required but missing properties.

        """
        ret = super().check_properties()
        if self.properties is None:
            warnings.warn("No available properties had been set, skipping check.")
            return None
        if ret:
            warnings.warn(f"Loaded bundle does not contain the following required properties: {ret}")
        # also check whether the optional properties use correct ID name if existing
        wrong_props = []
        for n, p in self.properties.items():
            if not p.get(BundleProperty.REQUIRED, False) and not self._check_optional_id(name=n, property=p):
                wrong_props.append(n)
        if wrong_props:
            warnings.warn(f"Loaded bundle defines the following optional properties with wrong ID: {wrong_props}")
        if ret is not None:
            ret.extend(wrong_props)
        return ret

    def _run_expr(self, id: str, **kwargs: dict) -> Any:
        return self.parser.get_parsed_content(id, **kwargs) if id in self.parser else None

    def _get_prop_id(self, name: str, property: dict) -> Any:
        prop_id = property[BundlePropertyConfig.ID]
        if prop_id not in self.parser:
            if not property.get(BundleProperty.REQUIRED, False):
                return None
            else:
                raise KeyError(f"Property '{name}' with config ID '{prop_id}' not in the config.")
        return prop_id

    def _get_property(self, name: str, property: dict) -> Any:
        """
        With specified property name and information, get the parsed property value from config.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        if not self._is_initialized:
            raise RuntimeError("Please execute 'initialize' before getting any parsed content.")
        prop_id = self._get_prop_id(name, property)
        return self.parser.get_parsed_content(id=prop_id) if prop_id is not None else None

    def _set_property(self, name: str, property: dict, value: Any) -> None:
        """
        With specified property name and information, set value for the expected property.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
            value: value to set for the property.

        """
        prop_id = self._get_prop_id(name, property)
        if prop_id is not None:
            self.parser[prop_id] = value
            # must parse the config again after changing the content
            self._is_initialized = False
            self.parser.ref_resolver.reset()

    def add_property(  # type: ignore[override]
        self, name: str, required: str, config_id: str, desc: str | None = None
    ) -> None:
        """
        Besides the default predefined properties, some 3rd party applications may need the bundle
        definition to provide additional properties for the specific use cases, if the bundle can't
        provide the property, means it can't work with the application.
        This utility adds the property for the application requirements check and access.

        Args:
            name: the name of target property.
            required: whether the property is "must-have".
            config_id: the config ID of target property in the bundle definition.
            desc: descriptions for the property.

        """
        super().add_property(name=name, required=required, desc=desc)
        self.properties[name][BundlePropertyConfig.ID] = config_id  # type: ignore[index]

    def _check_optional_id(self, name: str, property: dict) -> bool:
        """
        If an optional property has reference in the config, check whether the property is existing.
        If `ValidationHandler` is defined for a training workflow, will check whether the optional properties
        "evaluator" and "val_interval" are existing.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        id = property.get(BundlePropertyConfig.ID, None)
        ref_id = property.get(BundlePropertyConfig.REF_ID, None)
        if ref_id is None:
            # no ID of reference config item, skipping check for this optional property
            return True
        # check validation `validator` and `interval` properties as the handler index of ValidationHandler is unknown
        ref: str | None = None
        if name in ("evaluator", "val_interval"):
            if f"train{ID_SEP_KEY}handlers" in self.parser:
                for h in self.parser[f"train{ID_SEP_KEY}handlers"]:
                    if h["_target_"] == "ValidationHandler":
                        ref = h.get(ref_id, None)
        else:
            ref = self.parser.get(ref_id, None)
        # for reference IDs that not refer to a property directly but using expressions, skip the check
        if ref is not None and not ref.startswith(EXPR_KEY) and ref != ID_REF_KEY + id:
            return False
        return True

    @staticmethod
    def patch_bundle_tracking(parser: ConfigParser, settings: dict) -> None:
        """
        Patch the loaded bundle config with a new handler logic to enable experiment tracking features.

        Args:
            parser: loaded config content to patch the handler.
            settings: settings for the experiment tracking, should follow the pattern of default settings.

        """
        for k, v in settings["configs"].items():
            if k in settings["handlers_id"]:
                engine = parser.get(settings["handlers_id"][k]["id"])
                if engine is not None:
                    handlers = parser.get(settings["handlers_id"][k]["handlers"])
                    if handlers is None:
                        engine["train_handlers" if k == "trainer" else "val_handlers"] = [v]
                    else:
                        handlers.append(v)
            elif k not in parser:
                parser[k] = v
        # save the executed config into file
        default_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = parser.get("execute_config", None)
        if filepath is None:
            if "output_dir" not in parser:
                # if no "output_dir" in the bundle config, default to "<bundle root>/eval"
                parser["output_dir"] = f"{EXPR_KEY}{ID_REF_KEY}bundle_root + '/eval'"
            # experiment management tools can refer to this config item to track the config info
            parser["execute_config"] = parser["output_dir"] + f" + '/{default_name}'"
            filepath = os.path.join(parser.get_parsed_content("output_dir"), default_name)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        parser.export_config_file(parser.get(), filepath)
