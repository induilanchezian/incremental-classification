from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler 
from keras.backend.tensorflow_backend import set_session
from matplotlib import pyplot as plt
from keras import backend as K

import tensorflow as tf
import numpy as np
import math
import os
import glob
import argparse

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

class TestCallback(Callback):
  def __init__(self, current_model_layers, prev_model_layers, lambda_const):
    self.current_model_layers = current_model_layers
    self.prev_model_layers = prev_model_layers
    self.lambda_const = lambda_const
    
  def on_epoch_end(self, epoch, logs={}):
    l2_sum = 0
    for (layer1, layer2) in zip(self.prev_model_layers, self.current_model_layers):
      old_weights = layer1.get_weights()
      new_weights = layer2.trainable_weights
      if (old_weights and new_weights):
        l2_weights_diff = new_weights[0] - old_weights[0]
        l2_weights_diff_squared = tf.square(l2_weights_diff)
        l2_weights_sum = tf.reduce_sum(l2_weights_diff_squared)

        l2_bias_diff = new_weights[1] - old_weights[1]
        l2_bias_diff_squared = tf.square(l2_bias_diff)
        l2_bias_sum = tf.reduce_sum(l2_bias_diff_squared)

        l2_sum += l2_weights_sum
        l2_sum += l2_bias_sum
    l2_sum = (self.lambda_const/2) * l2_sum

    print('\nL2 difference after {} epoch: {}\n'.format(epoch, K.eval(l2_sum)))

def l2_loss(y_true, y_pred, prev_model_layers, curr_model_layers, lambda_const):
  l2_sum = 0 
  for (layer1, layer2) in zip(prev_model_layers, curr_model_layers):
    old_weights = layer1.get_weights()
    new_weights = layer2.get_weights()
    if (old_weights and new_weights):
      l2_weights_diff = new_weights[0] - old_weights[0] 
      l2_weights_diff_squared = tf.square(l2_weights_diff)
      l2_weights_sum = tf.reduce_sum(l2_weights_diff_squared)

      l2_bias_diff = new_weights[1] - old_weights[1]
      l2_bias_diff_squared = tf.square(l2_bias_diff)
      l2_bias_sum = tf.reduce_sum(l2_bias_diff_squared)

      l2_sum += l2_weights_sum 
      l2_sum += l2_bias_sum 
  l2_sum = (lambda_const/2) * l2_sum 
  loss = (K.binary_crossentropy(y_true, y_pred)) + l2_sum
  return loss

def step_decay(epoch):
   initial_lrate = 1e-3
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def train(train_data_dir, validation_data_dir, prev_model_file, epochs, batch_size, lambda_const, input_shape, weights_path, loss_file):
  nb_train_samples = np.sum([len(glob.glob(train_data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(train_data_dir)])
  nb_validation_samples = np.sum([len(glob.glob(validation_data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(validation_data_dir)])

  prev_model = load_model(prev_model_file)
  for layer in prev_model.layers:
      layer.trainable = False
  
  current_model = load_model(prev_model_file)
  outputTensor = current_model.output

  sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
  current_model.compile(loss=lambda y_true, y_pred: l2_loss(y_true, y_pred, prev_model.layers, current_model.layers, lambda_const), 
          optimizer=sgd,
          metrics=['accuracy'])

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

  checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  learning_rate = LearningRateScheduler(step_decay)
  history = LossHistory()
  callbacks_list = [checkpoint, learning_rate, history]

  current_model.fit_generator(
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
  parser.add_argument('--prev_model', type=str, help='Path to previous model checkpoint file')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')
  parser.add_argument('--batch_size', type=int, help='Training and Validation batch size')
  parser.add_argument('--reg_lambda', type=float, help='Regularization constant')
  parser.add_argument('--image_height', type=int, help='Image height crop size')
  parser.add_argument('--image_width', type=int, help='Image width crop size')
  parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
  parser.add_argument('--log_file', type=str, help='Path to log file')
 
  args = parser.parse_args()
  
  if K.image_data_format() == 'channels_first':
    input_shape = (3, args.image_width, args.image_height)
  else:
    input_shape = (args.image_width, args.image_height, 3)
  
  train(args.train_dir, args.validation_dir, args.prev_model, args.num_epochs, args.batch_size, args.reg_lambda, (args.image_width, args.image_height), 
        args.checkpoint_file, args.log_file)
 



