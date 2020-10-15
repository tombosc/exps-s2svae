#!/bin/bash

# Grid-search PreUni

DATASET="yahoo"
MODEL="lstm_max_uni" 
ROOT_DIR="$FWP_ROOT_EXP/exp_${DATASET}_preuni"
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

FINAL_MODEL="lstm_max" # CAREFUL!!!
# the code does NOT raise an error when the encoder of this model and the 
# pretrained model are incompatible.
# Be sure that they match, otherwise the pretrained model will be discarded.
# For example, "bow_max" goes with "bow_max_uni", "lstm" with "lstm_uni", ...

ROOT_DIR_PREUNI="$FWP_ROOT_EXP/exp_${DATASET}_uni"
for NZ in 4 16; do
    for SEED in 1 2 3; do
        for FREE_BITS in 2 8; do
			continue
            PRETRAINED_LM="$ROOT_DIR/fs_fb${FB_STYLE}_${FREE_BITS}_wu${WARMUP}_${MODEL}_nz${NZ}_sd${SEED}"
            ./wrap_from_pre_uni.sh $FREE_BITS $WARMUP $FB_STYLE $ROOT_DIR_PREUNI $DATASET $FINAL_MODEL $SEED "$PRETRAINED_LM/model.pt" $NZ
        done
    done
done
