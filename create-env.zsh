#! /bin/zsh

mamba env create -n pumt -f environment.yaml
mamba activate pumt
local env_file=$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=$PWD/third-party/LuoLib:\$PYTHONPATH" > $env_file
export BUILD_MONAI=1
echo "export BUILD_MONAI=1" >> $env_file
pip install --no-build-isolation -e third-party/MONAI
