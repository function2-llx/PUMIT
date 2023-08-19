# PUMT

## Create Environment

[Mamba](https://mamba.readthedocs.io) is recommended to manage virtual environments. 

```zsh
./create-env.zsh
```

## Run Visual Tokenizer Pre-Training

```zsh
python scripts/tokenizer/main.py -c conf/tokenizer/simple/main.yaml --data.dl_conf.train_batch_size 8 --data.dl_conf.num_workers 10 --training.benchmark true --model.quantize.mode soft --model.quantize.num_embeddings 1024 --loss.entropy_weight 1 --loss.quant_weight 0.03
```

Note that you may need to adjust the batch size according to the number of GPUs.

## Run ViT Pre-Training

```zsh
scripts/model/mim-b.zsh --data.dl_conf.train_batch_size 14 --data.dl_conf.num_workers 10 --model.tokenizer.path <tokenizer checkpoit path>
```

## 