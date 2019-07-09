#!/bin/bash

TRAIN_DIR=../../data/incremental_data/incremental_1
VAL_DIR=../../data/preprocessed_5c/val
TEST_DIR=../../data/preprocessed_5c/test
BASE_TRAIN_DIR=../../data/base_data/train

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
PREV_MODEL=../../models/base/multiclass/weights_multiclass_smaller_2.50-1.45.hdf5
CKPT_DIR=../../models/incremental/multiclass/weight_transfer
CKPT=weights_multiclass_smaller_transfer
LOG=losses_1.txt
DEVICE=0

REPORT=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-test.json
ROC=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-test.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-inc1-train.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-inc1-train.png

BASE_TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-base-train.json
BASE_TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-weight-transfer-base-train.png

#train using weight-transfer on incremental data
CUDA_VISIBLE_DEVICES=$DEVICE python incremental_weight_transfer.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --prev_model $PREV_MODEL --num_epochs $EPOCHS --batch_size $BATCH_SIZE --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT_DIR --checkpoint_file $CKPT --log_file $LOG

#evaluate model on held-out common test set
#python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#evaluate model on incremental training data (set 1)
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
#evaluate model on base training data 
#python evaluate.py --data_dir $BASE_TRAIN_DIR --model $CKPT --output_file $BASE_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $BASE_TRAIN_ROC


