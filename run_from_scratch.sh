#!/bin/bash

# Grid-search on VAEs without pretraining.

DATASET="yahoo"
MODEL="bow_max" # "BoW-max-LSTM"
ROOT_DIR="$FWP_ROOT_EXP/exp_${DATASET}_fs"
FB_STYLE=1 
WARMUP=0
for NZ in 4 16; do
    for FREE_BITS in 2 8; do
        for SEED in 1 2 3; do
			continue
            ./wrap_fs.sh $FREE_BITS $WARMUP $FB_STYLE $ROOT_DIR $DATASET $MODEL $SEED -1 $NZ
        done
    done
done
