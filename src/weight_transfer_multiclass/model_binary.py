from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant

def model(input_shape):
  model = Sequential()
  model.add(Cropping2D(cropping=(2,4),input_shape=input_shape))
  model.add(Conv2D(32, (7,7), strides=(2,2), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  #model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
  #          bias_initializer=Constant(0.1)))
  #model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
  #          bias_initializer=Constant(0.1)))
  #model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=Orthogonal(1.0),
  #          bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(32, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(32, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))


  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(128, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))
 
  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(Conv2D(256, (3, 3), strides=(1,1), padding='same', kernel_initializer=Orthogonal(1.0),
            bias_initializer=Constant(0.1)))
  model.add(LeakyReLU(0.5))

  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Dropout(0.5))
  model.add(GlobalMaxPooling2D())
  model.add(Dense(1, kernel_initializer=Orthogonal(1.0), bias_initializer=Constant(0.1), activation='sigmoid'))
  
  return model



