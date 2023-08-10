#!/usr/bin/env zsh

set -x

python scripts/downstream/chaos/main.py fit \
    -c conf/downstream/chaos/fit.yaml \
    --model.backbone conf/model/vit-b.yaml \
    --model.backbone conf/model/eva02-b-ckpt.yaml \
    $@
