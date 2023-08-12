#!/usr/bin/env zsh

set -x

python scripts/downstream/btcv/main.py test \
    -c conf/downstream/btcv/test.yaml \
    --model.backbone conf/model/vit-b.yaml \
    $@
