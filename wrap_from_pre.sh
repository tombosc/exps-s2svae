#!/bin/bash

# PreAE, phase 2:
# - reset the decoder's weights

export PYTHONIOENCODING=utf-8

FREE_BITS="$1"
WARMUP="$2"
FB_STYLE="$3"
ROOT_DIR="$4"
DATASET="$5"
BASE_MODEL="$6"
SEED="$7"
LOAD_PATH="$8"
ENCODE_LENGTH="$9" # ignored
NZ="${10}"

#echo $SLURM_JOB_ID

echo "All arguments: 1 $FREE_BITS 2 $WARMUP 3 $FB_STYLE 4 $ROOT_DIR 5 $DATASET 6 $BASE_MODEL 7 $SEED 8 $LOAD_PATH 9 $ENCODE_LENGTH 10 $NZ"

EXP_DIR="$ROOT_DIR/from_pre_fb${FB_STYLE}_${FREE_BITS}_warmup${WARMUP}_${BASE_MODEL}_nz${NZ}_sd${SEED}"

if [ "$WARMUP" -eq 0 ]; then
    KL_START=1
else
    KL_START=0
fi

python text_anneal_fb.py \
    --dataset $DATASET \
    --exp_dir $EXP_DIR \
    --load_path $LOAD_PATH \
    --reset_dec \
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
