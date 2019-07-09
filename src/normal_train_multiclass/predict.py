from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from base_train_multiclass_model2 import accuracy, maxout, maxout_shape
from scipy import interp
import os
import glob
import argparse
import json
import numpy as np 
import itertools
import cv2

def predict(image_file, model):
  img = cv2.imread(image_file)
  print(np.max(img))
  img = cv2.resize(img, dsize=(512,512), interpolation=cv2.INTER_NEAREST)
  img = np.expand_dims(img, axis=0)
  img = img / 255.0

  prediction = model.predict(img)
  pred = prediction[0]
  font = cv2.FONT_HERSHEY_SIMPLEX
  img = img[0,:,:,:] * 255
  if pred <= 0.5:
    cv2.putText(img,'mild',(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join('../../data/predictions/3', image_file.split('/')[-1].replace('jpeg', 'png')), img)  
  elif pred > 0.5 and pred <= 1.5:
    cv2.putText(img,'moderate',(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join('../../data/predictions/3', image_file.split('/')[-1].replace('jpeg', 'png')), img)  
  elif pred > 1.5 and pred <= 2.5:
    cv2.putText(img,'severe',(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join('../../data/predictions/3', image_file.split('/')[-1].replace('jpeg', 'png')), img)  
  else:
    cv2.putText(img,'proliferative',(10,300), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join('../../data/predictions/3', image_file.split('/')[-1].replace('jpeg', 'png')), img)  
  return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_file', type=str, help='Data directory on which to evaluate model')
  parser.add_argument('--model', type=str, help='Path to the checkpoint file of model to be evaluated')

  args = parser.parse_args()
  loaded_model = load_model(args.model, custom_objects={'accuracy': accuracy, 'maxout': maxout, 'maxout_shape': maxout_shape})
  ifiles = map(lambda x: os.path.join(args.image_file, x), os.listdir(args.image_file))
  for img_file in ifiles:
    predict(img_file, loaded_model)



