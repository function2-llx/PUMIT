#!/usr/bin/env zsh

set -x

python scripts/tokenizer/main.py \
    -c conf/tokenizer/simple/main-sharpen.yaml \
    --loss.quant_weight=-0.1 \
    --data.trans_conf.isotropic_th 1.5 \
    --training.pretrained_ckpt_path pre-trained/simple.ckpt \
    $@
