from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate, Lambda
from keras.layers import LeakyReLU, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant

import tensorflow as tf
import numpy as np

from custom_layers import maxout, maxout_shape

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
  model.add(Dense(512, kernel_initializer=Orthogonal(1.0), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  model.add(LeakyReLU(0.01))
  model.add(Lambda(lambda x : maxout(x, 128), lambda shp : maxout_shape(shp, 128)))
  model.add(Dense(1, kernel_initializer=Orthogonal(1.0), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.0005)))
  
  return model



