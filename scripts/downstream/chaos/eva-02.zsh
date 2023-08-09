#!/usr/bin/env zsh

set -x

python scripts/downstream/chaos/main.py fit \
    -c conf/downstream/chaos/fit.yaml \
    --data.num_workers 8 \
    --model.backbone conf/model/vit-b.yaml \
    --model.backbone.eva02_pretrained_path pre-trained/eva02_B_pt_in21k_p14.pt \
    --model.backbone.pretrained_pos_embed_shape "[16, 16]" \
    $@
