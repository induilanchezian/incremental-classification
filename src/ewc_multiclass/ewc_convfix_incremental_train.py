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
from keras import losses


import tensorflow as tf
import math 
import numpy as np
import os
import glob
import argparse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def maxout(inputs, num_units):
    return tf.contrib.layers.maxout(inputs, num_units)

def maxout_shape(input_shape, num_units):
    output_shape = list(input_shape)
    output_shape[-1] = num_units
    return output_shape

def accuracy(y_true, y_pred):
  y_pred = tf.clip_by_value(y_pred, 0.0, 3.0)
  y_pred = tf.round(y_pred)
  y_true = tf.to_int32(y_true)
  y_pred = tf.to_int32(y_pred)
  return K.mean(K.equal(y_true, y_pred))

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

def step_decay(epoch):
   initial_lrate = 1e-3
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def l2_diff(prev_layers, current_layers, y_pred, outputTensor):
  l2_sum = 0
  for (layer1, layer2) in zip(prev_layers[34:], current_layers[34:]):
    old_weights = layer1.get_weights()
    new_weights = layer2.get_weights()
    if (old_weights and new_weights):
      weights_grads = tf.gradients(tf.log(outputTensor), layer2.trainable_weights[0])
      squared_weights_grads = tf.square(weights_grads)
      weights_diff = new_weights[0] - old_weights[0]
      squared_weights_diff = tf.square(weights_diff)
      weights_fisher_reg = tf.multiply(squared_weights_grads, squared_weights_diff)
      fisher_reg_sum_weights = tf.reduce_sum(weights_fisher_reg)
 
      bias_grads = tf.gradients(tf.log(outputTensor), layer2.trainable_weights[1])
      squared_bias_grads = tf.square(bias_grads)
      bias_diff = new_weights[1] - old_weights[1]
      squared_bias_diff = tf.square(bias_diff)
      bias_fisher_reg = tf.multiply(squared_bias_grads, squared_bias_diff)
      fisher_reg_sum_bias = tf.reduce_sum(bias_fisher_reg)

      l2_sum += fisher_reg_sum_weights
      l2_sum += fisher_reg_sum_bias

  return l2_sum

def ewc_loss(y_true, y_pred, prev_model_layers, curr_model_layers, outputTensor, lambda_const):
   fisher_reg = l2_diff(prev_model_layers, curr_model_layers, y_pred, outputTensor)
   reg_term = (lambda_const/2) * fisher_reg
   loss = losses.mean_squared_error(y_true, y_pred) + reg_term
   return loss

def train(train_data_dir, validation_data_dir, prev_model_file, epochs, batch_size, lambda_const, input_shape, checkpoint_dir, weights_path, loss_file):
  nb_train_samples = np.sum([len(glob.glob(train_data_dir+'/'+i+'/*.jpeg')) for i in ['1','2','3','4']])
  nb_validation_samples = np.sum([len(glob.glob(validation_data_dir+'/'+i+'/*.jpeg')) for i in ['1','2','3','4']])

  prev_model = load_model(prev_model_file, custom_objects={'accuracy':accuracy, 'maxout': maxout, 'maxout_shape': maxout_shape})
  for layer in prev_model.layers:
    layer.trainable = False
  
  current_model = load_model(prev_model_file, custom_objects={'accuracy':accuracy, 'maxout': maxout, 'maxout_shape': maxout_shape})
  for layer in current_model.layers[:34]:
    layer.trainable = False
  outputTensor = current_model.output

  sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
  current_model.compile(loss=lambda y_true, y_pred: ewc_loss(y_true, y_pred, prev_model.layers, current_model.layers, outputTensor, lambda_const), 
          optimizer=sgd,
          metrics=[accuracy])

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
      classes=['1','2','3','4'])

  validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='sparse',
      classes=['1','2','3','4'])

  checkpoint = ModelCheckpoint(checkpoint_dir + '/' + weights_path + '.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
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
  parser.add_argument('--reg_const', type=float, help='Regularization constant')
  parser.add_argument('--image_height', type=int, help='Image height crop size')
  parser.add_argument('--image_width', type=int, help='Image width crop size')
  parser.add_argument('--checkpoint_dir', type=str, help='Relative path of the directory where checkpoints are stored')
  parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
  parser.add_argument('--log_file', type=str, help='Path to log file')
 
  args = parser.parse_args()
  
  if K.image_data_format() == 'channels_first':
    input_shape = (3, args.image_width, args.image_height)
  else:
    input_shape = (args.image_width, args.image_height, 3)
  
  train(args.train_dir, args.validation_dir, args.prev_model, args.num_epochs, args.batch_size, args.reg_const, (args.image_width, args.image_height), 
        args.checkpoint_dir, args.checkpoint_file, args.log_file)
 



