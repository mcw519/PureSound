#!/bin/bash

LIBRIMIX=$1
OUTPUT_FOLDER=$2
LIBRISPEECH=$3


# 16k-2mix-max-clean
dev_clean_mata=${LIBRIMIX}/storage/Libri2Mix/wav16k/max/metadata/mixture_dev_mix_clean.csv
python parser.py $dev_clean_mata ${OUTPUT_FOLDER}/dev-clean librispeech_metadata/dev-clean $LIBRISPEECH

test_clean_mata=${LIBRIMIX}/storage/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv
python parser.py $test_clean_mata ${OUTPUT_FOLDER}/test-clean librispeech_metadata/test-clean $LIBRISPEECH

train100_clean_meta=${LIBRIMIX}/storage/Libri2Mix/wav16k/max/metadata/mixture_train-100_mix_clean.csv
python parser.py $train100_clean_meta ${OUTPUT_FOLDER}/train-100-clean librispeech_metadata/train-clean-100 $LIBRISPEECH

train360_clean_meta=${LIBRIMIX}/storage/Libri2Mix/wav16k/max/metadata/mixture_train-360_mix_clean.csv
python parser.py $train360_clean_meta ${OUTPUT_FOLDER}/train-100-clean librispeech_metadata/train-clean-360 $LIBRISPEECH
