import numpy as np 
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

import cv2 as cv
import tensorflow as tf
import pandas as pd
import os 
import argparse
from scipy import misc
import keras.backend as K
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

#Loops over each directory in the training folder 
#Loops over every file in each subdirectory

from model1 import model as m1 


def read_and_process_image(fpath, shp):
    img = cv.imread(fpath,1)
    img = cv.resize(img, shp, interpolation=cv.INTER_NEAREST)
    img = img.astype(np.float32)
    img = img/(255.0)
    return np.expand_dims(img, axis=0)

def get_features(model, image):
    feats = model.predict(image)
    feats = feats.flatten()
    return feats

def get_logit_model(ckpt, input_shape):
    model = m1(input_shape)
    logits = model.layers[-1].output
    print(logits.shape)
    model = Model(model.input, logits)
    return model

def get_nlfeats_model(ckpt, input_shape):
    model = m1(input_shape)
    model.load_weights(ckpt) 
    model.pop()
    model.pop()
    model.pop()
    nl_feats = model.layers[-1].output
    print(nl_feats.shape)
    model = Model(model.input, nl_feats)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, help='Data directory')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint file name')
    parser.add_argument('--features_file', type=str, help='Path of file where features are stored. Filename with .npy extension')

    args = parser.parse_args()

    shp = (512, 512)
    image_width, image_height = shp   
    batch_size=1 
    if K.image_data_format() == 'channels_first':
        input_shape = (3, image_width, image_height)
    else:
        input_shape = (image_width, image_height, 3)
    
    train_dir = args.train_dir
    ckpt = args.checkpoint 
    subdirs = ['1','2','3','4']
 
    features = []
    
    model = get_nlfeats_model(ckpt, input_shape)

    for subdir in subdirs:
        print(subdir)
        label = int(subdir)
        subpath = os.path.join(train_dir, subdir)
        test_steps = len(os.listdir(subpath))
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(train_dir,
                target_size=(image_width, image_height),
                batch_size=batch_size,
                class_mode='sparse',
                classes=[subdir],
                shuffle=False)
        predictions = model.predict_generator(test_generator, steps=test_steps)
        #predictions = predictions.ravel()

        predictions = np.vstack(predictions)
        print(predictions.shape)
        np.save(args.features_file+'_'+str(label), predictions)

        features = []









  

