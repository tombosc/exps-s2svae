#!/bin/bash

# train a deterministic AE model (useful as a phase 1 for PreAE)

export PYTHONIOENCODING=utf-8
echo $HOSTNAME

NAME_MODEL="$1"
ROOT_DIR="$2"
DATASET="$3"
SEED="$4"
ENCODE_LENGTH="$5"
NZ="$6"

# ENCODE_LENGTH is ignored

EXP_DIR="$ROOT_DIR/ae_${NAME_MODEL}_nz${NZ}_sd${SEED}"

#echo "All arguments: 1 $NAME_MODEL 2 $ROOT_DIR 3 $DATASET 4 $SEED 5 $ENC_LENGTH 6 $NZ" 7 "$MODEL"

python text_anneal_fb.py \
    --dataset $DATASET \
    --exp_dir $EXP_DIR \
    --kl_start 0 \
    --warm_up 0 \
    --fb 0 \
    --model-type $NAME_MODEL \
    --opt sgd \
    --encode_length -1 \
    --lr 0.5 \
    --seed $SEED \
    --nz $NZ 
