#!/bin/bash

TRAIN_DIR=../../data/preprocessed_5c_original_dist/train
ORIG_TRAIN_DIR=../../data/preprocessed_5c_balanced/train
VAL_DIR=../../data/preprocessed_5c/val
TEST_DIR=../../data/preprocessed_5c/test
INCREMENTAL_TRAIN_DIR=../../data/incremental_data/incremental_1
DEVICE=0

EPOCHS_1=100
EPOCHS_2=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
CKPT_PATH=weights_multiclass_smaller_3 
CKPT=../../models/base/multiclass
CKPT_FILE=$CKPT/weights_multiclass_smaller.100-1.71.hdf5
LOG=losses_mc_1.txt

REPORT=modelv1-KaggleDRD-50ep-test-27_02_2019-balanced.json
ROC=modelv1-KaggleDRD-50ep-test-27_02_2019-balanced.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-train-27_02_2019-balanced.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-train-27_02_2019-balanced.png

INCREMENTAL_TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-train-27_02_2019-balanced.json
INCREMENTAL_TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-train-27_02_2019-balanced.png

CUDA_VISIBLE_DEVICES=$DEVICE python base_train_multiclass_model2.py --train_dir $ORIG_TRAIN_DIR --validation_dir $VAL_DIR --num_epochs $EPOCHS_2 --batch_size $BATCH_SIZE --learning_rate 0.001 --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT --checkpoint_file $CKPT_PATH --log_file $LOG 

#python base_train_multiclass_model2.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --num_epochs $EPOCHS_2 --batch_size $BATCH_SIZE --learning_rate 0.0001 --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT_PATH --checkpoint_file $CKPT --log_file $LOG --is_continue 1

#python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#python evaluate.py --data_dir $INCREMENTAL_DATA_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
#CUDA_VISIBLE_DEVICES=$DEVICE python evaluate.py --data_dir $INCREMENTAL_TRAIN_DIR --model $CKPT_FILE --output_file $INCREMENTAL_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $INCREMENTAL_TRAIN_ROC



