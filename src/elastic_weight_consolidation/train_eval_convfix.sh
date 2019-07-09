#!/bin/bash

TRAIN_DIR=../../data/incremental_data_binary/incremental_1
VAL_DIR=../../data/base_data/val
TEST_DIR=../../data/base_data/test
BASE_TRAIN_DIR=../../data/base_data/train

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
PREV_MODEL=../../models/base/modelv1-KaggleDRD-50ep-base.h5
CKPT=../../models/incremental/binary_ewc_convfix.h5
LOG=losses_1.txt

REPORT=modelv1-KaggleDRD-50ep-incremental1-ewc-test.json
ROC=modelv1-KaggleDRD-50ep-incremental1-ewc-test.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-ewc-inc1-train.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-ewc-inc1-train.png

BASE_TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-ewc-base-train.json
BASE_TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-ewc-base-train.png

LAMBDA=0.1

#train using ewc on incremental data
python ewc_convfix_incremental_train.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --prev_model $PREV_MODEL --num_epochs $EPOCHS --batch_size $BATCH_SIZE --reg_const $LAMBDA --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_file $CKPT --log_file $LOG

#evaluate model on held-out common test set
#python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#evaluate model on incremental training data (set 1)
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
#evaluate model on base training data 
#python evaluate.py --data_dir $BASE_TRAIN_DIR --model $CKPT --output_file $BASE_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $BASE_TRAIN_ROC


