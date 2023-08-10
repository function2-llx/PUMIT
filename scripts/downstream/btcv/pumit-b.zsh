#!/usr/bin/env zsh

set -x

python scripts/downstream/btcv/main.py fit \
    -c conf/downstream/btcv/fit.yaml \
    --model.backbone conf/model/vit-b.yaml \
    --model.backbone.pretrained_ckpt.path pre-trained/pumit-b.ckpt \
    $@
