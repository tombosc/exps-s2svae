#!/bin/bash

# Grid-search PreAE

DATASET="yahoo"
ROOT_DIR="$FWP_ROOT_EXP/exp_${DATASET}_baseline"

# Train deterministic AE for Li & al (2019)'s pretraining (PreAE in the paper)
MODEL="lstm_max" # lstm or lstm_max in the paper but can be whatever

for NZ in 4 16; do
    for SEED in 1 2 3; do
		continue
        # You can use slurm or any job manager...
        #sbatch --gres=gpu:1 --mem 10G -p main -t 3:00:00 wrap_vanilla_ae.sh $MODEL $ROOT_DIR $DATASET $SEED -1 $NZ
        # Or directly call
        ./wrap_vanilla_ae.sh $MODEL $ROOT_DIR $DATASET $SEED -1 $NZ
    done
done

# Then, reset decoder, retrain.

FB_STYLE=1
WARMUP=0
for NZ in 4 16; do
    for SEED in 1 2 3; do
        for FREE_BITS in 2 8; do
            continue
            EXP_DIR="$ROOT_DIR/ae_${MODEL}_nz${NZ}_sd${SEED}"
            ./wrap_from_pre.sh $FREE_BITS $WARMUP $FB_STYLE $ROOT_DIR $DATASET $MODEL $SEED "$EXP_DIR/model.pt" 1 $NZ
        done
    done
done

