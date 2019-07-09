import sys
import sys
sys.path.append('../')

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

def get_id(filename):
    return filename.split('.')[0]

#Split the filename into different tokens separated by _
#The number before the hyphen denotes the duplicate id
#The middle token indicates the patient id
#The last token can be 'left.jpeg' or 'right.jpeg'
def fname_split(filename):
    return filename.split('_')

#Takes a list of tokens as input
#Returns a tuple of elements
#The first element in the tuple specifies 'left' if
#image of lefet and 'right' otherwise
#The second element in the tuple is the patient id
def get_pid_lr(split_fname):
    lr = split_fname.pop()
    return lr, split_fname

def get_complementary_file(filename):
    split_fname = fname_split(filename)
    lr, pid = get_pid_lr(split_fname)
    complement_string = 'right.jpeg'
    if lr == 'left.jpeg':
        pid.append(complement_string)
        return '_'.join(pid)
    return None

def get_features(model, dataframe, data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_generator = datagen.flow_from_dataframe(dataframe, data_dir, target_size=(512,512), shuffle=False)
    feats = model.predict_generator(data_generator, steps=2)
    feats = feats.flatten()
    return feats

def get_feature_model(ckpt):
    model = load_model(ckpt, custom_objects={'accuracy': accuracy, 'maxout':maxout, 'maxout_shape':maxout_shape})
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
    data_generator = datagen.flow_from_dataframe(dataframe, data_dir, target_size=(512,512), shuffle=False)
    feats = model.predict_generator(data_generator, steps=2)
    feats = feats.flatten()
    return feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_csv', type=str, help='CSV format file which maps file ID to the label/diabetic level')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint file name')
    parser.add_argument('--features_file', type=str, help='Path of file where features are stored. Filename with .npy extension')
    #parser.add_argument('--labels_file', type=str, help='Path of file where labels are stored. Filename with .npy extension')

    args = parser.parse_args()

    data_dir = args.data_dir
    ckpt = args.checkpoint 

    #data_dir = '/home/indu/Thesis/incremental-classification/data/preprocessed_5c_original_dist/train' 
    #ckpt = '/home/indu/Thesis/incremental-classification/models/base/multiclass/weights_multiclass.100-1.96.hlabel_df5'
    
    label_df = pd.read_csv(args.label_csv)

    #labels = [] 
    features = []
    
    model = get_feature_model(ckpt)
    
    files_list = os.listdir(data_dir)
    files_list.sort() 
    for fname in files_list:
        comp_file = get_complementary_file(fname)
        #If comp_file returns None this is not done
        #Ignored for right files 
        if comp_file in files_list:
            left_id = get_id(fname)
            right_id = get_id(comp_file)
                
            left_label = label_df.loc[label_df['image']== left_id, 'level'].iloc[0]
            right_label = label_df.loc[label_df['image']== right_id, 'level'].iloc[0]

            label = np.maximum(left_label, right_label)
            print('FILE: {}, LABEL: {}'.format(fname, label))

            #if (label > 0):
            fpath = os.path.join(data_dir, fname)
            comp_path = os.path.join(data_dir, comp_file)
      
            d = {'filename': [fname, comp_file], 'class': [left_label, right_label]}
            temp_frame = pd.DataFrame(data=d)
      
            feats = get_features(model, temp_frame, data_dir)
            features.append(feats)
            #labels.append(label-1)
            #files_list.remove(comp_file)

    features = np.vstack(features)
    print(features.shape)
    np.save(args.features_file, features)
    #np.save(args.labels_file, labels)








  

