#!/bin/bash

TRAIN_DIR=/media/indu/New\ Volume/data/base_data/train
VAL_DIR=/media/indu/New\ Volume/data/base_data/val
TEST_DIR=/media/indu/New\ Volume/data/base_data/test
#INCREMENTAL_TRAIN_DIR=../../data/incremental_data_binary/incremental_1
#INCREMENTAL_TRAIN_DIR=../../data/incremental_data/incremental_1

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
#CKPT_PATH=../../models/joint
#CKPT=$CKPT_PATH/weights.40-0.40.hdf5
CKPT=/media/indu/New\ Volume/models/base/modelv4-KaggleDRD-50ep-base.h5
LOG=losses_2.txt

REPORT=modelv4-KaggleDRD-50ep-test.json
ROC=modelv4-KaggleDRD-50ep-test.png
DEVICE=0

#TRAIN_REPORT=modelv2-KaggleDRD-50ep-JOINT-train.json
#TRAIN_ROC=modelv2-KaggleDRD-50ep-JOINT-train.png

#INCREMENTAL_TRAIN_REPORT=modelv1-KaggleDRD-50ep-JOINT-incremental1-train.json
#INCREMENTAL_TRAIN_ROC=modelv1-KaggleDRD-50ep-JOINT-incremental1-train.png

#INCREMENTAL_TRAIN_REPORT=test.json
#INCREMENTAL_TRAIN_ROC=test.png

CUDA_VISIBLE_DEVICES=$DEVICE python base_train_binary.py --train_dir "$TRAIN_DIR" --validation_dir "$VAL_DIR" --num_epochs $EPOCHS --batch_size $BATCH_SIZE --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_file "$CKPT" --log_file $LOG
#CuDA_VISIBLE_DEVICES=$DEVICE python evaluate.py --data_dir "$TEST_DIR" --model "$CKPT" --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC

#python evaluate.py --data_dir $INCREMENTAL_TRAIN_DIR --model $CKPT --output_file $INCREMENTAL_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $INCREMENTAL_TRAIN_ROC



