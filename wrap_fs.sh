#!/bin/bash

# train a VAE model from scratch

export PYTHONIOENCODING=utf-8
echo $HOSTNAME

FREE_BITS="$1"
WARMUP="$2"
FB_STYLE="$3"
ROOT_DIR="$4"
DATASET="$5"
MODEL="$6"
SEED="$7"
ENCODE_LENGTH="${8}" # disabled
NZ="${9}"


EXP_DIR="$ROOT_DIR/fs_fb${FB_STYLE}_${FREE_BITS}_wu${WARMUP}_${MODEL}_nz${NZ}_sd${SEED}"

if [ "$WARMUP" -eq 0 ]; then
    KL_START=1
else
    KL_START=0
fi

python text_anneal_fb.py \
    --dataset $DATASET \
    --exp_dir $EXP_DIR \
    --kl_start $KL_START \
    --warm_up $WARMUP \
    --target_kl "$FREE_BITS" \
    --fb $FB_STYLE \
    --opt sgd \
    --lr 0.5 \
    --seed $SEED \
    --model-type $MODEL \
    --encode_length -1 \
    --nz $NZ 
