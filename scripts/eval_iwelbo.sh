#!/bin/bash

# compute perplexity and NLL using importance weighted bound

RUN_EVAL=0
ROOT_DIR="$FWP_ROOT_EXP"
DATASET="yahoo"
for MODEL_DIR in $ROOT_DIR/exp_${DATASET}_baseline/*; do
    OUTPUT_FILE="$MODEL_DIR/test_eval.txt"
    #if [ -e $OUTPUT_FILE ];then
    #    continue
    #fi

    if [ $RUN_EVAL -eq 1 ]; then
        MODEL="$MODEL_DIR/model.pt"
        python text_anneal_fb.py \
            --dataset $DATASET \
            --load-args-from-logs \
            --load_path $MODEL \
            --seed 1 \
            --iw_nsamples 500 \
            --eval > $OUTPUT_FILE
    fi
    if [ -e $OUTPUT_FILE ]; then
		echo $MODEL_DIR
		tail -n 1 $OUTPUT_FILE
	fi
done
