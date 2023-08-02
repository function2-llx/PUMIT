#!/usr/bin/env zsh

set -x

python scripts/model/main.py fit \
    -c conf/model/main.yaml \
    --model conf/model/vit-l.yaml \
    --model conf/model/mim-l.yaml \
    --model.eva02_pretrained_path pre-trained/eva02_L_pt_m38m_p14.pt \
    --model.tokenizer conf/tokenizer/simple/pre-trained.yaml \
    --data conf/model/cuda.yaml \
    --model.temperature 0.1 \
    $@
