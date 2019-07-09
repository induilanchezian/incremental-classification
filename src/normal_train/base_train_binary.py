from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from matplotlib import pyplot as plt
from keras import backend as K
from model_binary import model 

import tensorflow as tf
import numpy as np
import os
import glob
import argparse
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def train(model, train_data_dir, validation_data_dir, epochs, batch_size, input_shape, weights_path, loss_file):
  sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
  model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
  nb_train_samples = np.sum([len(glob.glob(train_data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(train_data_dir)])
  nb_validation_samples = np.sum([len(glob.glob(validation_data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(validation_data_dir)])

  img_width, img_height = input_shape

  # this is the augmentation configuration we will use for training
  train_datagen = ImageDataGenerator(
      rescale=1. / 255,
      zoom_range=0.2,
      brightness_range=[0.5,2.0],
      horizontal_flip=True)

  # this is the augmentation configuration we will use for testing:
  # only rescaling
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='binary')

  validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='binary')

  checkpoint = ModelCheckpoint(weights_path, save_best_only=True)
  learning_rate = LearningRateScheduler(step_decay)
  history = LossHistory()
  callbacks_list = [checkpoint, learning_rate, history]

  model.fit_generator(
      train_generator,
      steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=nb_validation_samples // batch_size,
      callbacks=callbacks_list
      )
  
  np.savetxt(loss_file, history.losses) 
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_dir', type=str, help='Path to training data directory')
  parser.add_argument('--validation_dir', type=str, help='Path to validation data directory')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')
  parser.add_argument('--batch_size', type=int, help='Training and Validation batch size')
  parser.add_argument('--image_height', type=int, help='Image height crop size')
  parser.add_argument('--image_width', type=int, help='Image width crop size')
  parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
  parser.add_argument('--log_file', type=str, help='Path to log file')
 
  args = parser.parse_args()
  
  if K.image_data_format() == 'channels_first':
    input_shape = (3, args.image_width, args.image_height)
  else:
    input_shape = (args.image_width, args.image_height, 3)
  
  fundus_model_binary = model(input_shape)
  train(fundus_model_binary, args.train_dir, args.validation_dir, args.num_epochs, args.batch_size, (args.image_width, args.image_height), 
        args.checkpoint_file, args.log_file)
 



