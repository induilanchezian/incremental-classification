import tensorflow as tf
import keras.backend as K

def accuracy(y_true, y_pred):
  y_pred = tf.clip_by_value(y_pred, 0.0, 3.0)
  y_pred = tf.round(y_pred)
  y_true = tf.to_int32(y_true)
  y_pred = tf.to_int32(y_pred)
  return K.mean(K.equal(y_true, y_pred))
