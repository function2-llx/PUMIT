# PUMT

## Create Environment

[Mamba](https://mamba.readthedocs.io) is recommended to manage virtual environments. 

```zsh
./create-env.zsh
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

Assume that the pre-trained checkpoint is placed at `./pre-trained/pumt.ckpt`.

### Classification

Execute scripts under `scripts/downstream/medmnistv2` for training and evaluation for each model.

### BTCV Segmentation

```zsh
scripts/downstream/btcv/pumit-b.zsh --data.num_workers 10 --data.ratio 1 --trainer.logger.name pumit-b --data.train_batch_size 4
```

### CHAOS Segmentation

```zsh
scripts/downstream/chaos/pumit-b.zsh --data.num_workers 10 --data.ratio 1 --trainer.logger.name pumit-b --data.train_batch_size 8
```
