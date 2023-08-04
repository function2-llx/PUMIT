python scripts/tokenizer/main.py \
    -c conf/tokenizer/simple/main.yaml \
    --data.dl_conf.num_workers 11 \
    --training.pretrained_ckpt_path pre-trained/simple.ckpt \
    --training.benchmark true \
    $@
