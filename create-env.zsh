#! /bin/zsh

mamba env create -n pumt -f environment.yaml
mamba activate pumt
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/MONAI
