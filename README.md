# PUMIT

This is the code repository for the paper _Pre-trained Universal Medical Image Transformer_ ([arxiv](https://arxiv.org/abs/2312.07630))

This repository keeps updating for further work. You may want to switch to the `submit` branch to reproduce the results from the paper, note that submodules need checkout as well.

## Create Environment

[Mamba](https://mamba.readthedocs.io) is recommended to manage virtual environments. 

```zsh
mamba env create -n pumit -f environment.yaml
mamba activate pumit
echo "export PYTHONPATH=$PWD:\$PYTHONPATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/MONAI
```

## Prepare data

Download datasets and put them under the `./datasets` folder, e.g.:

```zsh
datasets
├── AbdomenCT-1K
├── ACDC
├── AMOS22
├── BCV
├── BrainPTM-2021
├── BraTS2023
...
```

Then run pre-proecssing script: `python scripts/data/process.py <dataset name>`. More details can be found in the script.
## Run Visual Tokenizer Pre-Training

```zsh
python scripts/tokenizer/main.py -c conf/tokenizer/simple/main.yaml --data.dl_conf.train_batch_size 8 --data.dl_conf.num_workers 10 --training.benchmark true --model.quantize.mode soft --model.quantize.num_embeddings 1024 --loss.entropy_weight 1 --loss.quant_weight 0.03
```

Note that you may need to adjust the batch size according to the number of GPUs.

## Run ViT Pre-Training

```zsh
scripts/model/mim-b.zsh --data.dl_conf.train_batch_size 14 --data.dl_conf.num_workers 10 --model.tokenizer.path <tokenizer checkpoit path>
```

## Run Downstream Tasks

Assume that the pre-trained checkpoint is placed at `./pre-trained/pumit.ckpt`.

### Classification

Execute scripts under `scripts/downstream/medmnistv2` for training and evaluation for each model.

### BTCV Segmentation

Download the BTCV data from the [official challenge](https://www.synapse.org/#!Synapse:syn3193805), and download the train/validation split file from [SMIT's repository](https://github.com/The-Veeraraghavan-Lab/SMIT/blob/5d81399010dcf8c03a944544dc42b62603075b13/dataset/dataset_0.json), organize the files as following: 

```plain
data
├── BTCV
│   ├── smit.json
│   ├── Testing
│   └── Training
```

Then run fine-tuning and inference:

```zsh
scripts/downstream/btcv/pumit-b.zsh --data.num_workers 10 --data.ratio 1 --trainer.logger.name pumit-b --data.train_batch_size 4
scripts/downstream/btcv/test-b.zsh --data.num_workers 10 --ckpt_path <output checkpoint path> --trainer.logger.name pumit-b
```

### CHAOS Segmentation

First, run the pre-processing script to convert the DICOM series into NIFTI format: `python scripts/downstream/chaos/preprocess.py`

Then run fine-tuning and inference:

```zsh
scripts/downstream/chaos/pumit-b.zsh --data.num_workers 10 --data.ratio 1 --trainer.logger.name pumit-b --data.train_batch_size 8
scripts/downstream/chaos/predict-b.zsh --data.num_workers 10 --ckpt_path <output checkpoint path> --trainer.logger.name pumit-b
```
