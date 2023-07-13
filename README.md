# PUMT

[Mamba](https://mamba.readthedocs.io) is recommended to manage virtual environments. 

```zsh
git clone --recursive git@github.com:function2-llx/PUMT.git <dir>
cd <dir>
mamba env create -n <env> -f environment.yaml
mamba activate <env>
echo "export PYTHONPATH=$PWD/third-party/LuoLib:\$PYTHONPATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
