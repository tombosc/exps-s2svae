#!/bin/bash

# Pool results from subdirectories and analyze results 
# ./scripts/analyze.sh dataset_name [ pool_cv | pool_all | eval_ssl ]

shopt -s extglob
DS="$1"
POOL=$2
ARCHIVE_DIR="npy_archives"

if [ -z "$DS" ]; then
    echo "Error: no dataset supplied"
    exit
fi

if [ -z "$POOL" ]; then
    echo "Error: Use 0 to visualize results, 1 for pooling with limited train, 2 for pooling with all training"
    exit
fi

FILES=$FWP_ROOT_EXP/*${DS}*
echo $FILES
mkdir -p $ARCHIVE_DIR

if [ "$POOL" = "pool_cv" ]; then
    echo "Pooling SSL results for dataset $DS"
    for REGIME in 5 50 500 5000; do
        #python pool_results.py "$ARCHIVE_DIR/results_fs_samples_B_${DS}_${REGIME}.npy" \
        python pool_results.py "$ARCHIVE_DIR/results_fs_samples_B_${DS}_${REGIME}.npy" \
                        --aggregate $FILES --n_samples_per_class $REGIME --in_fn results_sample_1_$REGIME.json

    done
elif [ "$POOL" = "pool_all" ]; then
    python pool_results.py "$ARCHIVE_DIR/results_fs_samples_B_${DS}_all_linear.npy" \
                    --aggregate $FILES --n_samples_per_class "-1" --in_fn results_sample_all_linear.json
    python pool_results.py "$ARCHIVE_DIR/results_fs_samples_B_${DS}_all_mlp.npy" \
                    --aggregate $FILES --n_samples_per_class "-1" --in_fn results_sample_all_mlp.json
elif [ "$POOL" = "eval_ssl" ]; then
	# Semi-supervised learning evaluation (Sec. 5 of paper)
	# Outputs Table 1 rows (for 1 dataset!)
	python print_eval_ssl.py $ARCHIVE_DIR/results_fs_samples_B_${DS}_{5,all_mlp}.npy "5,all"
	# Outputs Table 8 rows (for 1 dataset!)
	#python print_eval_ssl.py $ARCHIVE_DIR/results_fs_samples_B_${DS}_{5,50,500,5000,all_linear}.npy "5,50,500,5000,all"
	# Also, you can compare linear and MLP classifiers using all data.
	#python print_eval_ssl.py $ARCHIVE_DIR/results_fs_samples_B_${DS}_{all_linear,all_mlp}.npy "lin,mlp"
elif [ "$POOL" = "eval_dec" ]; then
	# Text generation evaluation (Sec. 6 of paper)
	# Table 2: note that it does not matter which ".npy" you use:
	# they should all contain the same decoding information...
	python print_decoding_accuracy.py npy_archives/results_fs_samples_B_${DS}_5.npy "5"
else
	echo "Unrecognized option: $POOL"
fi

