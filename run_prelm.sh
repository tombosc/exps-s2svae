#!/bin/bash

# Grid-search PreLM

DATASET="yahoo"
ROOT_DIR="$FWP_ROOT_EXP/exp_${DATASET}_baseline"

# Train LSTM-LM

for SEED in 1 2 3; do
	continue
    ./wrap_lstm_lm.sh $DATASET $ROOT_DIR $SEED
done

# Our pretraining method (PreLM)

MODEL="lstm_avg" # in our experiments, lstm_max wa unstable :( 

WARMUP=0
FB_STYLE=1
ROOT_DIR_PRELM="$FWP_ROOT_EXP/exp_${DATASET}_prelm"
for NZ in 4 16; do
    for SEED in 1 2 3; do
        for FREE_BITS in 2 8; do
			continue
            PRETRAINED_LM="$ROOT_DIR/lm_baseline_seed${SEED}"
            ./wrap_from_pre_enc.sh $FREE_BITS $WARMUP $FB_STYLE $ROOT_DIR_PRELM $DATASET $MODEL $SEED "$PRETRAINED_LM/model.pt" 0 $NZ
        done
    done
done


