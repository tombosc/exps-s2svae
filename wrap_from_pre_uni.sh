#!/bin/bash

# Pretraining Unigram "PreUni", phase 2:
# - entire encoder is frozen including linear transformations for computing variational parameters
# - uses --ignore_load_errors to use a different decoder without an error. This means that a problem in loading the encoder will also not raise an error! Be extra careful.

export PYTHONIOENCODING=utf-8

FREE_BITS="$1"
WARMUP="$2"
FB_STYLE="$3"
ROOT_DIR="$4"
DATASET="$5"
BASE_MODEL="$6"
SEED="$7"
LOAD_PATH="$8"
NZ="${9}"


EXP_DIR="$ROOT_DIR/from_pre_fb${FB_STYLE}_${FREE_BITS}_warmup${WARMUP}_${BASE_MODEL}_nz${NZ}_sd${SEED}_freezenc"

if [ "$WARMUP" -eq 0 ]; then
    KL_START=1
else
    KL_START=0
fi

python text_anneal_fb.py \
    --ignore_load_errors \
    --dataset $DATASET \
    --exp_dir $EXP_DIR \
    --load_path $LOAD_PATH \
    --model-type $BASE_MODEL \
    --kl_start $KL_START \
    --warm_up $WARMUP \
    --freeze_encoder \
    --target_kl "$FREE_BITS" \
    --fb $FB_STYLE \
    --opt sgd \
    --lr 0.5 \
    --seed $SEED \
    --nz $NZ 

# In this setting, the encoder is frozen so we can copy all the results
# of SSL directly.
PRETRAINED_PATH=$(dirname $LOAD_PATH)
cp "$PRETRAINED_PATH/results_"*.json $EXP_DIR
