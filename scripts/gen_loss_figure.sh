#!/bin/sh

# Generate loss figures (Fig. 1, 2, 3, 4 in the paper) and save in $OUT_FN
# Requires calling scripts/compute_loss_per_pos.sh first
# ./scripts/gen_loss_figure.sh dataset_name

DATASET=$1
EXPS="$FWP_ROOT_EXP/exp_$DATASET*"
OUT_FN="loss_${DATASET}_long.pdf"
python gen_loss_figure.py $EXPS $OUT_FN
