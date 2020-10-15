#!/bin/bash

# PreLM, phase 2:
# - reset the decoder's weights
# - freeze the encoder's weights, but NOT the linear transformations (which were not trained during phase 1 at all!)

export PYTHONIOENCODING=utf-8

FREE_BITS="$1"
WARMUP="$2"
FB_STYLE="$3"
ROOT_DIR="$4"
DATASET="$5"
BASE_MODEL="$6"
SEED="$7"
LOAD_PATH="$8"
ENCODE_LENGTH="$9"
NZ="${10}"

EXP_DIR="$ROOT_DIR/from_pre_enc_fb${FB_STYLE}_${FREE_BITS}_warmup${WARMUP}_${BASE_MODEL}_nz${NZ}_sd${SEED}"

if [ "$WARMUP" -eq 0 ]; then
    KL_START=1
else
    KL_START=0
fi

echo "All arguments: 1 $FREE_BITS 2 $WARMUP 3 $FB_STYLE 4 $ROOT_DIR 5 $DATASET 6 $BASE_MODEL 7 $SEED 8 $LOAD_PATH 9 $ENCODE_LENGTH 10 $NZ"

python text_anneal_fb.py \
    --dataset $DATASET $OPTION \
    --exp_dir $EXP_DIR \
    --reset_dec \
    --freeze_encoder_exc \
    --load_encoder $LOAD_PATH \
    --kl_start $KL_START \
    --warm_up $WARMUP \
    --target_kl "$FREE_BITS" \
    --fb $FB_STYLE \
    --encode_length -1 \
    --opt sgd \
    --lr 0.5 \
    --model-type $BASE_MODEL \
    --seed $SEED \
    --nz $NZ 
