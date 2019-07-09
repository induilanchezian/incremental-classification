import sys
sys.path.append('../')

import numpy as np 
import argparse

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, Lambda
from keras.initializers import Orthogonal, glorot_normal 
from keras.regularizers import l2
from keras.optimizers import SGD 

from base_train_multiclass_model2 import accuracy, maxout, maxout_shape
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from matplotlib import pyplot as plt
import itertools

from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
    
  print(cm)
    
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def model(input_shape):
    inputs = Input(shape=input_shape)
    # a layer instance is callable on a tensor, and returns a tensor
    hidden1 = Dense(64, activation='relu', kernel_initializer=glorot_normal(20), kernel_regularizer=l2(0.001))(inputs)
    maxout1 = Lambda(lambda x : maxout(x, 32), lambda shp : maxout_shape(shp, 32))(hidden1)
    hidden2 = Dense(64, activation='relu', kernel_initializer=glorot_normal(seed=20), kernel_regularizer=l2(0.001))(maxout1)
    maxout2 = Lambda(lambda x : maxout(x, 32), lambda shp : maxout_shape(shp, 32))(hidden2)
    predictions = Dense(1, kernel_initializer=glorot_normal(seed=20), kernel_regularizer=l2(0.001))(maxout2)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    return model

def save_predictions(model, X_test, y_left, y_right, predictions_file):
    y_test = np.append(y_left, y_right)
    
    predictions = model.predict(X_test)
    predictions = predictions.ravel()
    predictions = np.append(predictions, predictions)
    y_pred = np.clip(predictions, 0.0, 4.0)
    y_pred[y_pred<=0.5] = 0
    y_pred[np.logical_and(y_pred>0.5, y_pred <=1.5)] = 1
    y_pred[np.logical_and(y_pred>1.5, y_pred <=2.5)] = 2
    y_pred[y_pred>=2.5] = 3
    np.save(predictions_file, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_feature_file', type=str, help='.npy file containing test features')
    parser.add_argument('--test_labels_left_file', type=str, help='.npy file containing test_labels')
    parser.add_argument('--test_labels_right_file', type=str, help='.npy file containing test_labels')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to store the model')
    parser.add_argument('--predictions', type=str, help='.npy file to store the predictions')

    args = parser.parse_args()

    X_test = np.load(args.test_feature_file)
    y_left = np.load(args.test_labels_left_file)
    y_right = np.load(args.test_labels_right_file)

    mdl = model((X_test.shape[1],))
    mdl.load_weights(args.checkpoint)
    save_predictions(mdl, X_test, y_left, y_right, args.predictions)





