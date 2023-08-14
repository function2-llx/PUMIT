#!/usr/bin/env zsh

set -x

python scripts/downstream/chaos/main.py predict \
    -c conf/downstream/chaos/predict.yaml \
    --model.backbone conf/model/vit-b.yaml \
    --model.sw_overlap 0.75 \
    $@
