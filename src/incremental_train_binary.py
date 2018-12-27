from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from matplotlib import pyplot as plt
from keras import backend as K

import numpy as np
import os
import glob
import argparse
import tensorflow as tf

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class IncrementalModel:
  def __init__(self, train_dir, validation_dir, epochs, batch_size, 
               img_width, img_height, checkpoint, loss_file, config_file):
    self.train_dir = train_dir
    self.validation_dir = validation_dir
    self.checkpoint_path = checkpoint
    self.num_epochs = epochs
    self.batch_size = batch_size
    self.config_file = config_file
    self.img_width = img_width 
    self.img_height = img_height
    self.loss_file = loss_file 

    if K.image_data_format() == 'channels_first':
      self.input_shape = (3, self.img_width, self.img_height)
    else:
      self.input_shape = (self.img_width, self.img_height, 3)

  def get_models_list(self):
    self.models = np.load_txt(self.config_file)

  def get_base_model(self):
    self.base_model = models[0]

  def get_latest_model(self):
    self.latest_model = models[-1]
  
  def previous_model(self):
    self.prev_model = load_model(self.latest_model)
    for layer in self.prev_model.layers:
      layer.trainable = false 

  def add_current_model(self):
    self.models.append(self.checkpoint_path)
    np.savetxt(config_file, self.models)

  def l2_weight_diff(self):
    for (layer1, layer2) in (self.model_old.layers, self.model.layers):
      old_weights = self.model_old.layer1.get_weights()[0]
      new_weights = self.model.layer2.get_weights()[0]
      l2_sum += tf.square(old_weights - new_weights)
    return l2_sum 

  def incremental_loss(self):
    def loss(y_true, y_pred):
      return K.binary_cross_entropy(y_true, y_pred) + l2_weight_diff()
    return loss

  def train(self):
    nb_train_samples = np.sum([len(glob.glob(self.train_dir+'/'+i+'/*.jpeg')) for i in os.listdir(self.train_dir)])
    nb_validation_samples = np.sum([len(glob.glob(self.validation_dir+'/'+i+'/*.jpeg')) for i in os.listdir(self.validation_dir)])
    
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    self.model = load_model(self.checkpoint_path)
    self.model.compile(loss=self.incremental_loss(),
              optimizer=sgd,
              metrics=['accuracy'])

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
        self.train_dir,
        target_size=(self.img_width, self.img_height),
        batch_size=self.batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        self.validation_dir,
        target_size=(self.img_width, self.img_height),
        batch_size=self.batch_size,
        class_mode='binary')

    checkpoint = ModelCheckpoint(incremental_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-4)
    history = LossHistory()
    callbacks_list = [checkpoint, reduce_lr, history]

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // self.batch_size,
        epochs=self.epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // self.batch_size,
        callbacks=callbacks_list
        )
  
    #print(history.losses)
    np.savetxt(self.loss_file, history.losses) 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_dir', type=str, help='Path to training data directory')
  parser.add_argument('--validation_dir', type=str, help='Path to validation data directory')
  parser.add_argument('--num_epochs', type=int, help='Number of training epochs') 
  parser.add_argument('--batch_size', type=int, help='Training and validation batch size')
  parser.add_argument('--image_width', type=int, help='Image height')
  parser.add_argument('--image_height', type=int, help='Image width')
  parser.add_argument('--checkpoint_file', type=str, help='Path to model checkpoint file')
  parser.add_argument('--log_file', type=str, help='Log file')
  parser.add_argument('--config', type=str, help='Configuration file for incremental training')

  incremental_model = IncrementalModel(args.train_dir, args.validation_dir, args.num_epochs, args.batch_size, 
                                       args.image_width, args.img_height, checkpoint_file, log_file, config_file)
  
  incremental_model.train()  
  incremental_model.add_current_model()
 
   


	



