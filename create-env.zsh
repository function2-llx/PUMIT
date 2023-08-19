#! /bin/zsh

mamba env create -n pumit -f environment.yaml
mamba activate pumit
echo "export PYTHONPATH=$PWD:\$PYTHONPATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/MONAI
