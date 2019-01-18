#!/bin/bash

TRAIN_DIR=../../data/incremental_data_binary/incremental_1
VAL_DIR=../../data/base_data/val
TEST_DIR=../../data/base_data/test

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
PREV_MODEL=../../models/base/modelv1-KaggleDRD-50ep-base.h5
CKPT=../../models/incremental/modelv1-KaggleDRD-50ep-incremental1-distillation.h5
LOG=losses_1.txt

REPORT=modelv1-KaggleDRD-50ep-incremental1-distillation-test.json
ROC=modelv1-KaggleDRD-50ep-incremental1-distillation-test.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-distillation-train.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-distillation-train.png

python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
