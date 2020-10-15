#!/bin/bash

# Semi-supervised learning evaluation
# (script takes no argument)
# modify the for loop to iterate on your experiment subdirectories

RUN_EVAL=0 # 1: run evaluation and print results, 0: only print
ROOT_DIR="$FWP_ROOT_EXP"
DATASET="amazon"
for MODEL_DIR in $ROOT_DIR/exp_${DATASET}_baseline/from_pre_fb1_8_warmup0_lstm_max_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_baseline/from_pre_fb1_8_warmup0_lstm_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_fs/fs_fb1_8_wu0_bow_max_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_uni/from_pre_fb1_8_warmup0_lstm_max_el-1_nz16_sd*_freezenc \
				 $ROOT_DIR/exp_${DATASET}_uni/from_pre_fb1_8_warmup0_no_enc_el-1_nz16_sd*_freezenc \
				 $ROOT_DIR/exp_${DATASET}_prelm/from_pre_enc_fb1_8_warmup0_lstm_pool_avg_el-1_nz16_sd* \
	; do
    if [ $RUN_EVAL -eq 1 ]; then
		MODEL="$MODEL_DIR/model.pt"
		if [ ! -f "$MODEL" ]; then
			echo "MODEL NOT FOUND" 
			continue
		fi

        # Cross-validation over several data-regimes
        for N in 5 50 500 5000; do
			# When we have more than 1000 samples per class, it is useless to
			# do repeated k-fold
            if [ $N -ge 1000 ]; then
                REPEAT=1
            else
                REPEAT=10
            fi
            if [ ! -f "$MODEL_DIR/results_sample_1_$N.json" ]; then
                python eval_classification.py \
                    --dataset $DATASET \
                    --seed 1 \
                    --load-args-from-logs \
                    --load_path $MODEL \
                    --num_label_per_class $N \
                    --classify_using_samples \
                    --n_repeats $REPEAT \
                    --resample 1
				exit
            fi
        done
        # Full dataset with early-stopping on the validation set
        for CLF in mlp; do 
            OUTPUT="$MODEL_DIR/results_sample_all_${CLF}.json"
            if [ ! -f $OUTPUT ]; then
                echo "$CLF"
                python text_ss_ft.py \
                    --dataset $DATASET \
                    --freeze_enc \
                    --load_path $MODEL \
                    --seed 1 \
                    --opt sgd \
                    --load-args-from-logs \
                    --batch_size 256 \
                    --lr 0.1 \
                    --one-indexed-labels \
                    --discriminator ${CLF}
                tail -n 1 $MODEL_DIR/log_classifier_full_data_${CLF}.txt > $OUTPUT
            fi
        done
        echo $MODEL
        cat $MODEL_DIR/log_classifier.txt | tail -n 3 | head -n 1
    else
        echo $MODEL_DIR
        cat $MODEL_DIR/log_classifier.txt | tail -n 3 | head -n 1
    fi
done
