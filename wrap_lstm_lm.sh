#!/bin/bash

# train a LSTM model from scratch

export PYTHONIOENCODING=utf-8

DATASET=$1
ROOT_DIR=$2
SEED=$3

echo $DATASET $ROOT_DIR $SEED
EXP_DIR=$ROOT_DIR/lm_baseline_seed${SEED}

python lm.py \
    --exp_dir $EXP_DIR \
    --dataset $DATASET \
    --opt 'sgd' \
    --cuda \
    --seed $SEED \
    --lr 0.5
