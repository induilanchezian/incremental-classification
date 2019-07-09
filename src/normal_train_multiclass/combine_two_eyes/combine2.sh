#!/bin/bash

model1=/media/indu/New\ Volume/models/base/multiclass/weights_multiclass.100-1.96.hdf5
model2=/media/indu/New\ Volume/models/base/multiclass/weights_multiclass.90-1.96.hdf5
model3=/media/indu/New\ Volume/models/base/multiclass/weights_multiclass_v2.100-2.01.hdf5
model4=/media/indu/New\ Volume/models/base/multiclass/weights_multiclass_v2.95-2.00.hdf5
model5=/media/indu/New\ Volume/models/base/multiclass/weights_multiclass_smaller_2.50-1.45.hdf5

models=( "$model1" "$model2" "$model3" "$model4" "$model5" )

#trainfeats=( train_feats1.npy train_feats2.npy train_feats3.npy train_feats4.npy train_feats5.npy )
testfeats=( test_feats1.npy test_feats2.npy test_feats3.npy test_feats4.npy test_feats5.npy )
preds=( predictions_bal1.npy predictions_bal2.npy predictions_bal3.npy predictions_bal4.npy predictions_bal5.npy )
checkpoint=( combine_model_bal1.hdf5 combine_model_bal2.hdf5 combine_model_bal3.hdf5 combine_model_bal4.hdf5 combine_model_bal5.hdf5 )

trainLabels=train_labels.npy
test_labels=test_labels.npy
left=left_test.npy
right=right_test.npy
device=1

for i in 0 1 2
	do

		#CUDA_VISIBLE_DEVICES=$device python combine_features_v2.py --data_dir ../../../data/preprocessed_train --label_csv ../../../data/trainLabels.csv --checkpoint ${models[$i]} --features_file ${trainfeats[$i]} 
		#CUDA_VISIBLE_DEVICES=$device python combine_features_v2.py --data_dir "/media/indu/New Volume/data/test" --label_csv ../../../retinopathy_solution.csv --checkpoint "${models[$i]}" --features_file ${testfeats[$i]} 
	
		CUDA_VISIBLE_DEVICES=$device python predict_combine.py --test_feature_file ${testfeats[$i]} --test_labels_left_file $left --test_labels_right_file $right --checkpoint ${checkpoint[$i]} --predictions ${preds[$i]}
	done
