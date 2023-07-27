#!/usr/bin/env zsh

python scripts/model/main.py fit \
    -c conf/model/main.yaml \
    -c conf/model/main-b.yaml \
    --model conf/model/vit-b.yaml \
    --model.eva02_pretrained_path pre-trained/eva02_B_pt_in21k_p14.pt \
    --model.tokenizer conf/tokenizer/vqvae.yaml