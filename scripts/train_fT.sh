#!/bin/bash

DIM="100"
CLF_DIRNAME="fastText_classifiers"

mkdir -p $CLF_DIRNAME

train_sets=(
    "datasets/ag_news/train.txt"
    "datasets/amazon/train.txt.100k"
    "datasets/yahoo/train.txt.100k"
    "datasets/short_yelp_data/short_yelp.train.txt"
)

test_sets=(
    "datasets/ag_news/test.txt"
    "datasets/amazon/test.txt.10k"
    "datasets/yahoo/test.txt.10k"
    "datasets/short_yelp_data/short_yelp.test.txt"
)

dataset_names=(
    "agnews"
    "amazon"
    "yahoo"
    "yelp"
)

for I in $(seq 0 3); do
    TRAIN=${train_sets[$I]} 
    TEST=${test_sets[$I]} 
    DATASET_NAME=${dataset_names[$I]}
    echo $DATASET_NAME $TRAIN $TEST

    sed -e 's/^/__label__/' $TRAIN > $TRAIN.fT 
    sed -e 's/^/__label__/' $TEST > $TEST.fT 
    CLF_FN="$CLF_DIRNAME/model_${DATASET_NAME}_fT"
    echo $TRAIN.fT $CLF_FN
    fasttext supervised -input $TRAIN.fT -dim $DIM -minCount 15 -output $CLF_FN -epoch 5
    fasttext test $CLF_FN.bin $TEST.fT

    TRAIN_FIRST_3=${TRAIN}.first_3.fT
    TEST_FIRST_3=${TEST}.first_3.fT
    cat $TRAIN.fT | cut -d' ' -f1-3 > $TRAIN_FIRST_3
    cat $TEST.fT | cut -d' ' -f1-3 > $TEST_FIRST_3
    #head -n 3 ${TRAIN_FIRST_3}
    CLF_3_FN="$CLF_DIRNAME/model_${DATASET_NAME}_first3_fT"
    fasttext supervised -input $TRAIN_FIRST_3 -dim $DIM -wordNgrams 1 -minCount 15 -output $CLF_3_FN -epoch 5
    fasttext test $CLF_3_FN.bin $TEST_FIRST_3
done
