#!/usr/bin/env zsh

set -x

python scripts/tokenizer/main.py \
    -c conf/tokenizer/simple/main-sharpen.yaml \
    --model.quantize.temperature 0.1 \
    --data.trans_conf.isotropic_th 1.5 \
    $@
