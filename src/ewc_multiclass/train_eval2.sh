#!/bin/bash

TRAIN_DIR=../../data/incremental_data/incremental_1
VAL_DIR=../../data/preprocessed_5c/val
TEST_DIR=../../data/incremental_data/incremental_5
BASE_TRAIN_DIR=../../data/preprocessed_5c_original_dist/train

EPOCHS=50
BATCH_SIZE=64
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
PREV_MODEL=../../models/base/multiclass/weights_multiclass_v2.95-2.00.hdf5
CKPT_DIR=../../models/incremental/multiclass
WEIGHTS_PATH=weights_multiclass_v2_ewc95
LOG=losses_1.txt

REPORT=multiclass-v2-ewc-test-report.json
ROC=multiclass-v2-ewc-test-ROC.png

TRAIN_REPORT=multiclass-v2-ewc-train-report.json
TRAIN_ROC=multiclass-v2-ewc-train-ROC.png

BASE_TRAIN_REPORT=multiclass-v2-ewc-inc-train-report.json
BASE_TRAIN_ROC=multiclass-v2-ewc-inc-train-ROC.png

LAMBDA=0.1
DEVICE=1

#train using ewc on incremental data
CUDA_VISIBLE_DEVICES=$DEVICE python ewc_incremental_train.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --prev_model $PREV_MODEL --num_epochs $EPOCHS --batch_size $BATCH_SIZE --reg_const $LAMBDA --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT_DIR --checkpoint_file $WEIGHTS_PATH --log_file $LOG

#evaluate model on held-out common test set
#python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#evaluate model on incremental training data (set 1)
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
#evaluate model on base training data 
#python evaluate.py --data_dir $BASE_TRAIN_DIR --model $CKPT --output_file $BASE_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $BASE_TRAIN_ROC


