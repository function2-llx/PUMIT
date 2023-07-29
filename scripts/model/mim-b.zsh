#!/usr/bin/env zsh

set -x

python scripts/model/main.py fit \
    -c conf/model/main.yaml \
    --model conf/model/vit-b.yaml \
    --model conf/model/mim-b.yaml \
    --model.eva02_pretrained_path pre-trained/eva02_B_pt_in21k_p14.pt \
    --model.tokenizer conf/tokenizer/simple/pretrained.yaml \
    --data conf/model/cuda.yaml \
    $@
