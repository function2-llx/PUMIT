#!/usr/bin/env zsh

python scripts/model/main.py fit \
    -c conf/model/main.yaml \
    --model conf/model/vit-l.yaml \
    --model conf/model/mim-l.yaml \
    --model.eva02_pretrained_path pre-trained/eva02_L_pt_m38m_p14.pt \
    --model.tokenizer conf/tokenizer/simple/pretrained.yaml
