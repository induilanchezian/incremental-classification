import numpy as np
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

temperature=2.0
lambda_const = 1.0

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.1
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

# now model outputs 512 dimensional vectors

def knowledge_distillation_loss(y_true, y_pred, lambda_const):
    y_true, logits = y_true[:, :1], y_true[:, 1:]
    y_soft = K.sigmoid(logits/temperature)
    y_pred, y_pred_soft = y_pred[:, :1], y_pred[:, 1:]
    return lambda_const*K.binary_crossentropy(y_true, y_pred) + 4.0*K.binary_crossentropy(y_soft, y_pred_soft)

def accuracy(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, :1]
    return binary_accuracy(y_true, y_pred)

if __name__ == '__main__':
    data_dir = '../../data/base_data/'
    train_dir = '../../data/incremental_data_binary/incremental_1'
    weights_path = '../../models/incremental/incremental1-cnn-fundus-distillation2.h5'

    nb_train_samples = np.sum([len(glob.glob(train_dir+'/'+i+'/*.jpeg')) for i in os.listdir(data_dir+'train')])
    nb_validation_samples = np.sum([len(glob.glob(data_dir+'val/'+i+'/*.jpeg')) for i in os.listdir(data_dir+'val')])
    epochs = 20
    batch_size = 64

    train_logits = np.load(data_dir + 'train_logits.npy')[()]
    val_logits = np.load(data_dir + 'val_logits.npy')[()]

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
