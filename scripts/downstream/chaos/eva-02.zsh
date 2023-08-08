#!/usr/bin/env zsh

python scripts/downstream/chaos/main.py fit \
    -c conf/downstream/chaos/main.yaml \
    --data.num_workers 8 \
    --model.backbone conf/model/vit-b.yaml \
    --eva02_pretrained_path pre-trained/eva02_B_pt_in21k_p14.pt \
