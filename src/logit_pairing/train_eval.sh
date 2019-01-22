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
CKPT=../../models/incremental/modelv1-KaggleDRD-50ep-incremental1-prob_pairing.h5
LOG=losses_1.txt

REPORT=modelv1-KaggleDRD-50ep-incremental1-prob_pairing-test.json
ROC=modelv1-KaggleDRD-50ep-incremental1-prob_pairing-test.png

TRAIN_REPORT=modelv1-KaggleDRD-50ep-incremental1-prob_pairing-train.json
TRAIN_ROC=modelv1-KaggleDRD-50ep-incremental1-prob_pairing-train.png

BASE_TRAIN_REPORT=modelv1-KaggleDRD-50ep-base-prob_pairing-train.json
BASE_TRAIN_ROC=modelv1-KaggleDRD-50ep-base-prob_pairing-train.png

TRAIN_LOGITS_FILE=../../data/incremental_data_binary/modelv1-50ep-logits-inc1.npy
VAL_LOGITS_FILE=../../data/base_data/modelv1-50ep-logits-val-base.npy
TEST_LOGITS_FILE=../../data/base_data/modelv1-50ep-logits-test-base.npy
BASE_TRAIN_LOGITS_FILE=../../data/base_data/modelv1-50ep-logits-train-base.npy

#training data logits
#python get_prev_model_logits.py --data_dir $TRAIN_DIR --logits_file $TRAIN_LOGITS_FILE --prev_model $PREV_MODEL
#validation data logits
#python get_prev_model_logits.py --data_dir $VAL_DIR --logits_file $VAL_LOGITS_FILE --prev_model $PREV_MODEL

#train using prob_pairing on incremental data
#python train_incremental_prob_pairing.py --train_dir $TRAIN_DIR --validation_dir $VAL_DIR --prev_model $PREV_MODEL --num_epochs $EPOCHS --batch_size $BATCH_SIZE --image_height $IMAGE_HEIGHT --image_width $IMAGE_WIDTH --checkpoint_file $CKPT --log_file $LOG --train_logits_file $TRAIN_LOGITS_FILE --val_logits_file $VAL_LOGITS_FILE

#Test data logits 
#python get_prev_model_logits.py --data_dir $TEST_DIR --logits_file $TEST_LOGITS_FILE --prev_model $PREV_MODEL
#Base training data logits
#python get_prev_model_logits.py --data_dir $BASE_TRAIN_DIR --logits_file $BASE_TRAIN_LOGITS_FILE --prev_model $PREV_MODEL

#evaluate model on held-out common test set
python evaluate.py --data_dir $TEST_DIR --logits_file $TEST_LOGITS_FILE --model $CKPT --output_file $REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $ROC
#evaluate model on incremental training data (set 1)
python evaluate.py --data_dir $TRAIN_DIR --logits_file $TRAIN_LOGITS_FILE --model $CKPT --output_file $TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $TRAIN_ROC
#evaluate model on base training data 
python evaluate.py --data_dir $BASE_TRAIN_DIR --logits_file $BASE_TRAIN_LOGITS_FILE --model $CKPT --output_file $BASE_TRAIN_REPORT --batch_size $BATCH_SIZE --image_width $IMAGE_WIDTH --image_height $IMAGE_HEIGHT --roc_file $BASE_TRAIN_ROC


