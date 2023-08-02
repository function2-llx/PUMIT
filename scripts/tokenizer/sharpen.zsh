#!/usr/bin/env zsh

set -x

python scripts/tokenizer/main.py \
    -c conf/tokenizer/simple/main.yaml \
    --model.vq_kwargs.temperature 0.1 \
    --data.trans_conf.isotropic_th 1.5 \
    $@
