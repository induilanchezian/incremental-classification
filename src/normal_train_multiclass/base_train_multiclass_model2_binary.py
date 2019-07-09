from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate, Lambda
from keras.layers import LeakyReLU, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from matplotlib import pyplot as plt
from keras import backend as K

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
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 100.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def maxout(inputs, num_units):
    return tf.contrib.layers.maxout(inputs, num_units)

def maxout_shape(input_shape, num_units):
    output_shape = list(input_shape)
    output_shape[-1] = num_units
    return output_shape

def model(input_shape):
  model = Sequential()
  model.add(Cropping2D(cropping=(2,4),input_shape=input_shape))
  model.add(Conv2D(32, (5,5), strides=(2,2), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))
  
  model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  #model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
  #          bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  #model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=Orthogonal(1.0),
  #          bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(64, (5, 5), strides=(2,2), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(512, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))
 
  model.add(Conv2D(512, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))

  model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(128, kernel_initializer=Orthogonal(1.0), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))
  model.add(Lambda(lambda x : maxout(x, 32), lambda shp : maxout_shape(shp, 32)))
  model.add(Dense(1, kernel_initializer=Orthogonal(1.0), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  
  return model

def accuracy(y_true, y_pred):
  y_pred = tf.clip_by_value(y_pred, 0.0, 1.4)
  y_pred = tf.round(y_pred)
  y_true = tf.to_int32(y_true)
  y_pred = tf.to_int32(y_pred)
  return K.mean(K.equal(y_true, y_pred))

def train(model, train_data_dir, validation_data_dir, epochs, batch_size, learning_rate, input_shape, weights_path, checkpoint_dir, loss_file):
  sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error',
                optimizer=sgd,
                metrics=[accuracy])
  nb_train_samples = np.sum([len(glob.glob(train_data_dir+'/'+i+'/*.jpeg')) for i in ['0','1']])
  nb_validation_samples = np.sum([len(glob.glob(validation_data_dir+'/'+i+'/*.jpeg')) for i in ['0','1']])

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
      class_mode='sparse',
      classes=['0','1'])

  validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      shuffle=False,
      class_mode='sparse',
      classes=['0','1'])

  checkpoint = ModelCheckpoint(checkpoint_dir + '/' + weights_path + '.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
  #learning_rate = LearningRateScheduler(step_decay)
  history = LossHistory()
  callbacks_list = [checkpoint,history]

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
  parser.add_argument('--learning_rate', type=float, help='Learning rate')
  parser.add_argument('--image_height', type=int, help='Image height crop size')
  parser.add_argument('--image_width', type=int, help='Image width crop size')
  parser.add_argument('--checkpoint_dir', type=str, help='Path to directory of model checkpoint file')
  parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
  parser.add_argument('--log_file', type=str, help='Path to log file')
  parser.add_argument('--continue_model', type=str, default=None, help='Whether to continue training from the given checkpoint file')
 
  args = parser.parse_args()
  
  if K.image_data_format() == 'channels_first':
    input_shape = (3, args.image_width, args.image_height)
  else:
    input_shape = (args.image_width, args.image_height, 3)
  
  if (args.continue_model is not None):
      model = load_model(args.continue_model, custom_objects={'accuracy':accuracy, 'maxout': maxout, 'maxout_shape': maxout_shape})
  else:
    model = model(input_shape)
  train(model, args.train_dir, args.validation_dir, args.num_epochs, args.batch_size, args.learning_rate, (args.image_width, args.image_height), 
        args.checkpoint_file, args.checkpoint_dir, args.log_file)
 



