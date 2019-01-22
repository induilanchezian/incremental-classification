import numpy as np
import argparse
import sys
import os
import glob
import math
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint

# use non standard flow_from_directory
from image_processing_v2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception

from keras.models import Model, load_model
from keras.layers import Lambda, concatenate, Activation, Dense
from keras.metrics import binary_accuracy
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

temperature=1.0
lambda_const = 0.01

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

def step_decay(epoch):
   initial_lrate = 1e-3
   drop = 0.1
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def prob_pairing_loss(y_true, y_pred, lambda_const):
    y_true, logits = y_true[:, :1], y_true[:, 1:]
    y_soft = K.sigmoid(logits/temperature)
    y_pred, y_pred_soft = y_pred[:, :1], y_pred[:, 1:]
    diff = y_soft - y_pred_soft
    squared_diff = tf.square(diff)
    return K.binary_crossentropy(y_true, y_pred) + lambda_const*K.sum(squared_diff)

def accuracy(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, :1]
    return binary_accuracy(y_true, y_pred)

def train(train_data_dir, validation_data_dir, prev_model, epochs, batch_size, input_shape, weights_path, loss_file, train_logits, val_logits):
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
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
      train_data_dir,train_logits,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,val_logits,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='binary')
    
    model = load_model(prev_model)
    dense_weights = model.layers[-1].get_weights()
    model.pop()
    features = model.layers[-1].output
    dense_output = Dense(1, weights=dense_weights)(features)
    model = Model(model.input, dense_output)
    
    # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation('sigmoid')(logits)
    
    # softed probabilities
    logits_T = Lambda(lambda x: x/temperature)(logits)
    probabilities_T = Activation('sigmoid')(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model.input, output)
    model.compile(
          optimizer=sgd,
          loss=lambda y_true, y_pred: prob_pairing_loss(y_true, y_pred, lambda_const),
          metrics=[accuracy]
     )

    checkpoint = ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, help='Path to training data directory')
    parser.add_argument('--validation_dir', type=str, help='Path to validation data directory')
    parser.add_argument('--prev_model', type=str, help='Path to previous model checkpoint file')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, help='Training and Validation batch size')
    parser.add_argument('--image_height', type=int, help='Image height crop size')
    parser.add_argument('--image_width', type=int, help='Image width crop size')
    parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--train_logits_file', type=str, help='Path to training data logits matrix of previous model')
    parser.add_argument('--val_logits_file', type=str, help='Path to validation data logits matrix of previous model')

    args = parser.parse_args()

    if K.image_data_format() == 'channels_first':
        input_shape = (3, args.image_width, args.image_height)
    else:
        input_shape = (args.image_width, args.image_height, 3)


    train_logits = np.load(args.train_logits_file)[()]
    val_logits = np.load(args.val_logits_file)[()]

    train(args.train_dir, args.validation_dir, args.prev_model, args.num_epochs, args.batch_size, (args.image_width, args.image_height),
         args.checkpoint_file, args.log_file, train_logits, val_logits)

    '''
    data_dir = '../../data/base_data/'
    train_dir = '../../data/incremental_data_binary/incremental_1'
    weights_path = '../../models/incremental/incremental1-cnn-fundus-distillation2.h5'

    nb_train_samples = np.sum([len(glob.glob(train_dir+'/'+i+'/*.jpeg')) for i in os.listdir(data_dir+'train')])
    nb_validation_samples = np.sum([len(glob.glob(val_dir+'/'+i+'/*.jpeg')) for i in os.listdir(data_dir+'val')])
    epochs = 20
    batch_size = 64


    train_datagen = ImageDataGenerator(
          rescale=1. / 255,
          zoom_range=0.2,
          horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

   # note: i'm also passing dicts of logits
    train_generator = train_datagen.flow_from_directory(
        train_dir, train_logits,
        target_size=(512, 512),
        batch_size=64,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        data_dir + 'val', val_logits,
        target_size=(512, 512),
        batch_size=64,
        class_mode='binary'
    )

    model = load_model('../../models/base/cnn-fundus.h5')
    dense_weights = model.layers[-1].get_weights()
    model.pop()
    features = model.layers[-1].output
    dense_output = Dense(1, weights=dense_weights)(features)
    model = Model(model.input, dense_output)

   # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation('sigmoid')(logits)

   # softed probabilities
    logits_T = Lambda(lambda x: x/temperature)(logits)
    probabilities_T = Activation('sigmoid')(logits_T)

    output = concatenate([probabilities, probabilities_T])
    model = Model(model.input, output)
    model.compile(
          optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
          loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
          metrics=[accuracy]
     )

    checkpoint = ModelCheckpoint(weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
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
    
    loss_file = 'loss_distillation_incremental_1.txt'
    np.savetxt(loss_file, history.losses)
    '''
