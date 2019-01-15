#!/bin/bash

TRAIN_DIR=../../data/base_data/train
VAL_DIR=../../data/base_data/val
TEST_DIR=../../data/base_data/test

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
CKPT=../../models/base/modelv1-KaggleDRD-50ep-base.h5
LOG=losses_1.txt

REPORT=modelv1-KaggleDRD-50ep-test.json
ROC=modelv1-KaggleDRD-50ep-test.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-train.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-train.png

#python base_train_binary.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --num_epochs $EPOCHS --batch_size $BATCH_SIZE --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_file $CKPT --log_file $LOG

python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC



