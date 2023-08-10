#!/usr/bin/env zsh

set -x

python scripts/downstream/chaos/main.py fit \
    -c conf/downstream/chaos/fit.yaml \
    --model.backbone conf/model/vit-b.yaml \
    $@
