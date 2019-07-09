import numpy as np 
from keras.models import Model, load_model
from base_train_multiclass_model2 import accuracy, maxout, maxout_shape
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import pandas as pd
import os 
import argparse
from scipy import misc
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

#Loops over each directory in the training folder 
#Loops over every file in each subdirectory

def fname_split(filename):
    return filename.split('_')

def get_pid_lr(split_fname):
    lr = split_fname.pop()
    return lr, split_fname

def get_complement_string(left_or_right):
    if left_or_right == 'left.jpeg':
        return 'right.jpeg'
    elif left_or_right == 'right.jpeg':
        return 'left.jpeg'
    else:
        return None

def get_complementary_file(filename):
    split_fname = fname_split(filename)
    lr, pid = get_pid_lr(split_fname)
    complement_string = get_complement_string(lr)
    pid.append(complement_string)
    return '_'.join(pid)

def get_features(model, dataframe, data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_generator = datagen.flow_from_dataframe(dataframe, data_dir, target_size=(512,512), shuffle=False)
    feats = model.predict_generator(data_generator, steps=2)
    feats = feats.flatten()
    return feats

def get_feature_model(ckpt):
    model = load_model(ckpt, custom_objects={'accuracy': accuracy, 'maxout':maxout, 'maxout_shape':maxout_shape})
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    nl_feats = model.layers[-1].output
    print(nl_feats.shape)
    model = Model(model.input, nl_feats)
    return model

def get_prediction_model(ckpt):
    model = load_model(ckpt, custom_objects={'accuracy': accuracy, 'maxout':maxout, 'maxout_shape':maxout_shape})
    logit = model.layers[-1].output
    model = Model(model.input, logit)
    return model

def get_preds(model, dataframe, data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_generator = datagen.flow_from_dataframe(dataframe, data_dir, target_size=(512,512), class_mode='sparse', shuffle=False)
    predictions = model.predict_generator(data_generator, steps=1)
    predictions = predictions.ravel()
    
    y_pred = np.clip(predictions, 0.0, 4.0)
    y_pred[y_pred<=0.5] = 0
    y_pred[np.logical_and(y_pred>0.5, y_pred <=1.5)] = 1
    y_pred[np.logical_and(y_pred>1.5, y_pred <=2.5)] = 2
    y_pred[y_pred>=2.5] = 3
    
    return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint file name')
    parser.add_argument('--features_file', type=str, help='Path of file where features are stored. Filename with .npy extension')
    parser.add_argument('--labels_file', type=str, help='Path of file where true labels are stored. Filename with .npy extension')

    args = parser.parse_args()

    data_dir = args.data_dir
    ckpt = args.checkpoint 

    #train_dir = '/home/indu/Thesis/incremental-classification/data/preprocessed_5c_original_dist/train' 
    #ckpt = '/home/indu/Thesis/incremental-classification/models/base/multiclass/weights_multiclass.100-1.96.hdf5'
    
    labels = [] 
    features = []
    
    model = get_prediction_model(ckpt)
    df = pd.read_csv('../../data/retinopathy_solution.csv')

    files_list = os.listdir(data_dir)
    for fname in files_list:
        print(fname)
        orig_name = fname.split('.')[0]
        comp_file = get_complementary_file(fname)
        comp_name = comp_file.split('.')[0]
        if comp_file in files_list:
            fpath = os.path.join(data_dir, fname)
            comp_path = os.path.join(data_dir, comp_file)
      
            label_orig = df.loc[df['image']== orig_name, 'level'].iloc[0] 
            label_comp = df.loc[df['image']== comp_name, 'level'].iloc[0]
            d = {'filename': [fname, comp_file], 'class': [label_orig, label_comp]}
            temp_frame = pd.DataFrame(data=d)

            true_labels = [label_orig, label_comp]
      
            feats = get_preds(model, temp_frame, data_dir)
            features.append(feats)
            labels.append(true_labels)
            files_list.remove(comp_file)

    features = np.vstack(features)
    labels = np.vstack(labels)
    print(features.shape)
    print(labels.shape)
    np.save(args.features_file, features)
    np.save(args.labels_file, labels)








  

