#!/bin/bash

TRAIN_DIR=../../data/preprocessed_5c/train
ORIG_TRAIN_DIR=../../data/preprocessed_5c_balanced/train
VAL_DIR=../../data/preprocessed_5c/val
TEST_DIR=../../data/preprocessed_5c/test
INCREMENTAL_TRAIN_DIR=/media/indu/New\ Volume/data/test_classes
DEVICE=1

EPOCHS_1=100
EPOCHS_2=50
BATCH_SIZE=54
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
CKPT_PATH1=weights_multiclass
CKPT_PATH2=weights_multiclass_v2
CKPT=/media/indu/New\ Volume/models/base/multiclass
CKPT_FILE01=/media/indu/New\ Volume/models/base/modelv1-KaggleDRD-50ep-base.h5
CKPT_FILE02=/media/indu/New\ Volume/models/base/modelv2-KaggleDRD-50ep-base.h5
CKPT_FILE03=/media/indu/New\ Volume/models/base/modelv3-KaggleDRD-50ep-base.h5
CKPT_FILE1=$CKPT/$CKPT_PATH1.100-1.96.hdf5
CKPT_FILE2=$CKPT/$CKPT_PATH2.100-2.01.hdf5
CKPT_FILE3=$CKPT/$CKPT_PATH1.90-1.96.hdf5
CKPT_FILE4=$CKPT/$CKPT_PATH2.95-2.00.hdf5
CKPT_FILE5=$CKPT/weights_multiclass_smaller_2.50-1.45.hdf5
CKPT_STRING=$CKPT_FILE01,$CKPT_FILE02,$CKPT_FILE03,$CKPT_FILE1,$CKPT_FILE2,$CKPT_FILE3,$CKPT_FILE4,$CKPT_FILE5

LOG=losses_mc_1.txt

REPORT=modelv1-KaggleDRD-50ep-test-14_03_2019-balanced.json
ROC=modelv1-KaggleDRD-50ep-test-14_03_2019-balanced.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-train-14_03_2019-balanced.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-train-14_03_2019-balanced.png

INCREMENTAL_TRAIN_REPORT=3_5class_ensemble_complete_test.json
INCREMENTAL_TRAIN_ROC=3_5class_ensemble_complete_test.png

#CUDA_VISIBLE_DEVICES=$DEVICE python base_train_multiclass_model2.py --train_dir $ORIG_TRAIN_DIR --validation_dir $VAL_DIR --num_epochs $EPOCHS_1 --batch_size $BATCH_SIZE --learning_rate 0.001 --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT --checkpoint_file $CKPT_PATH --log_file $LOG --is_continue 0

#python base_train_multiclass_model2.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --num_epochs $EPOCHS_2 --batch_size $BATCH_SIZE --learning_rate 0.0001 --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_dir $CKPT_PATH --checkpoint_file $CKPT --log_file $LOG --is_continue 1

#python evaluate.py --data_dir $TEST_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#python evaluate.py --data_dir $INCREMENTAL_DATA_DIR --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#python evaluate.py --data_dir $TRAIN_DIR --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
CUDA_VISIBLE_DEVICES=$DEVICE python evaluate_complete_with_ensemble.py --data_dir "$INCREMENTAL_TRAIN_DIR" --model "$CKPT_STRING" --output_file $INCREMENTAL_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $INCREMENTAL_TRAIN_ROC
#CUDA_VISIBLE_DEVICES=$DEVICE python evaluate_complete_with_ensemble.py --data_dir $TEST_DIR --model $CKPT_STRING --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC



