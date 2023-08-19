# MONAI

* **apps**: high level medical domain specific deep learning applications.

* **auto3dseg**: automated machine learning (AutoML) components for volumetric image analysis.

* **bundle**: components to build the portable self-descriptive model bundle.

* **config**: for system configuration and diagnostic output.

* **csrc**: for C++/CUDA extensions.

* **data**: for the datasets, readers/writers, and synthetic data.

* **engines**: engine-derived classes for extending Ignite behaviour.

* **fl**: federated learning components to allow pipeline integration with any federated learning framework.

* **handlers**: defines handlers for implementing functionality at various stages in the training process.

* **inferers**: defines model inference methods.

* **losses**: classes defining loss functions, which follow the pattern of `torch.nn.modules.loss`.

* **metrics**: defines metric tracking types.

* **networks**: contains network definitions, component definitions, and Pytorch specific utilities.

* **optimizers**: classes defining optimizers, which follow the pattern of `torch.optim`.

* **transforms**: defines data transforms for preprocessing and postprocessing.

* **utils**: generic utilities intended to be implemented in pure Python or using Numpy,
and not with Pytorch, such as namespace aliasing, auto module loading.

* **visualize**: utilities for data visualization.

* **_extensions**: C++/CUDA extensions to be loaded in a just-in-time manner using `torch.utils.cpp_extension.load`.
