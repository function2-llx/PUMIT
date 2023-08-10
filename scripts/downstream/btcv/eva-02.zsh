#!/usr/bin/env zsh

set -x

python scripts/downstream/btcv/main.py fit \
    -c conf/downstream/btcv/fit.yaml \
    --model.backbone conf/model/vit-b.yaml \
    --model.backbone.pretrained_ckpt conf/model/eva02-b-ckpt.yaml \
    --model.backbone.pretrained_pos_embed_shape "[16, 16]" \
    --data.num_workers 8 \
    $@
