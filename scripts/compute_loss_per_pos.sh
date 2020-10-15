#!/bin/bash

# Precompute losses for figures (Fig. 1, 2, 3, 4 in the paper)
# ./scripts/compute_loss_per_pos.sh dataset_name

DATASET="$1"
DOC_SIZES="8,16" # compute losses on docs of size 8 and 16 (paper uses 16 only)

# First, compute loss for baseline
if [ -z "$DATASET" ]; then
    echo "Error: no dataset supplied"
    exit
fi

for MODEL_DIR in $FWP_ROOT_EXP/exp_${DATASET}_baseline/lm_baseline_*; do
	echo $MODEL_DIR
    if [ -f $MODEL_DIR/"val_avg_loss_per_ts.npy" ]; then
		echo "already computed"
        continue
    fi
    python lm.py \
        --export-avg-loss-per-ts $DOC_SIZES \
        --load_path $MODEL_DIR/model.pt \
        --dataset $DATASET 
done

for MODEL_DIR in $FWP_ROOT_EXP/exp_${DATASET}_baseline/from_pre_fb1_2_warmup0_lstm_el-1_nz16* \
				 $FWP_ROOT_EXP/exp_${DATASET}_fs/fs_fb1_2_wu0_lstm_max_el-1_nz16* \
				 $FWP_ROOT_EXP/exp_${DATASET}_baseline/ae_lstm_el-1_nz16* \
; do
    echo "$MODEL_DIR"
    if [ -f $MODEL_DIR/"val_avg_loss_per_ts.npy" ]; then
		echo "already computed"
        continue
    fi

    python text_anneal_fb.py \
        --export-avg-loss-per-ts $DOC_SIZES \
        --load-args-from-logs \
        --load_path $MODEL_DIR/model.pt \
        --dataset $DATASET 
done
