#!/bin/bash

export PYTHONIOENCODING=utf-8

CLF_DIRNAME="fastText_classifiers"
N_SAMPLES=10000
RUN_EVAL=0
ROOT_DIR="$FWP_ROOT_EXP"
DATASET="yahoo"

for MODEL_DIR in $ROOT_DIR/exp_${DATASET}_baseline/from_pre_fb1_8_warmup0_lstm_max_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_baseline/from_pre_fb1_8_warmup0_lstm_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_fs/fs_fb1_8_wu0_bow_max_el-1_nz16_sd* \
				 $ROOT_DIR/exp_${DATASET}_uni/from_pre_fb1_8_warmup0_lstm_max_el-1_nz16_sd*_freezenc \
				 $ROOT_DIR/exp_${DATASET}_uni/from_pre_fb1_8_warmup0_no_enc_el-1_nz16_sd*_freezenc \
				 $ROOT_DIR/exp_${DATASET}_prelm/from_pre_enc_fb1_8_warmup0_lstm_pool_avg_el-1_nz16_sd* \
	; do
    echo $MODEL_DIR
    CLF_FN="$CLF_DIRNAME/model_${DATASET}_fT".bin
	for STRATEGY in beam greedy; do
		DECODED="$MODEL_DIR/decoded_test_${STRATEGY}"
		if [ $RUN_EVAL -eq 1 ]; then
			# reconstruct documents
            python text_anneal_fb.py \
                    --dataset $DATASET \
                    --load-args-from-logs \
                    --reconstruct_from $MODEL_DIR/model.pt \
                    --reconstruct_to $DECODED \
                    --reconstruct_add_labels_to_source \
                    --reconstruct_batch_size 1 \
                    --reconstruct_max_examples $N_SAMPLES \
                    --no-unk \
                    --decoding_strategy $STRATEGY > $MODEL_DIR/decoded_f1_${STRATEGY}_BLEU
			# compute agreement by using pretrained fasttext classifiers (run scripts/train_fT.sh first!)
            python f1_decoded_gt.py $CLF_FN \
                $DECODED{.hyp,.ref,.label} $MODEL_DIR/decoded_f1_${STRATEGY}.json
			# compute exact reconstruction rates (first word and length)
			python compute_exact_reconstruction_rates.py $DECODED.{hyp,ref} > ${DECODED}_stats.json
		fi
		# output results
		cat ${DECODED}_stats.json
	done
done
