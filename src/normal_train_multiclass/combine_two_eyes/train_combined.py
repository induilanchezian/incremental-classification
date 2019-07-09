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

def train(model, X, y, X_val, y_val, checkpoint, num_epochs=20):
    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd,
            loss='mean_squared_error',
            metrics=[accuracy])
    training_generator = BalancedBatchGenerator(X, y, sampler=RandomOverSampler(), batch_size=64, random_state=42)
    model.fit_generator(generator=training_generator, epochs=num_epochs, validation_data=(X_val,y_val))
    model.save_weights(checkpoint)

def evaluate(model, X_test, y_left, y_right, predictions_file):
    y_test = np.append(y_left, y_right)
    
    predictions = model.predict(X_test)
    predictions = predictions.ravel()
    predictions = np.append(predictions, predictions)
    predictions = predictions[y_test != 0]
    y_test = y_test[y_test != 0]
    y_test = y_test - 1

    y_pred = np.clip(predictions, 0.0, 4.0)
    y_pred[y_pred<=0.5] = 0
    y_pred[np.logical_and(y_pred>0.5, y_pred <=1.5)] = 1
    y_pred[np.logical_and(y_pred>1.5, y_pred <=2.5)] = 2
    y_pred[y_pred>=2.5] = 3

    y_pred = y_pred.astype(int)
    y_test = y_test.astype(int)
    print(y_pred)
    print(y_test)

    np.save(predictions_file, y_pred)

    kappa_score = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = [0,1,2,3]
    y_pred = label_binarize(y_pred, classes=classes)
    true_labels = label_binarize(y_test, classes=classes)

    num_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
 
    report = {}
    report['Accuracy'] = accuracy 
    report['AUC1'] = roc_auc[0]
    report['AUC2'] = roc_auc[1]
    report['AUC3'] = roc_auc[2]
    report['AUC4'] = roc_auc[3]
    report['AUC_Macro'] = roc_auc["macro"]
    report['AUC_Micro'] = roc_auc["micro"]
    report['kappa'] = kappa_score

    print('Accuracy: %.2f'%report['Accuracy'])
    print('AUC1: %.2f'%report['AUC1'])
    print('AUC2: %.2f'%report['AUC2'])
    print('AUC3: %.2f'%report['AUC3'])
    print('AUC4: %.2f'%report['AUC4'])
    print('AUC (micro): %.2f'%report['AUC_Micro'])
    print('AUC (macro): %.2f'%report['AUC_Macro'])
    print('Kappa score: %.2f'%report['kappa'])

    plt.figure()
    plot_confusion_matrix(cm, classes=classes, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_feature_file', type=str, help='.npy file containing training features')
    parser.add_argument('--train_labels_file', type=str, help='.npy file containing labels')
    parser.add_argument('--test_feature_file', type=str, help='.npy file containing test features')
    parser.add_argument('--test_labels_file', type=str, help='.npy file containing test_labels')
    parser.add_argument('--test_labels_left_file', type=str, help='.npy file containing test_labels')
    parser.add_argument('--test_labels_right_file', type=str, help='.npy file containing test_labels')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to store the model')
    parser.add_argument('--predictions', type=str, help='.npy file to store the predictions')

    args = parser.parse_args()

    X_train = np.load(args.train_feature_file)
    y_train = np.load(args.train_labels_file)

    X_test = np.load(args.test_feature_file)
    y_test = np.load(args.test_labels_file)
    y_left = np.load(args.test_labels_left_file)
    y_right = np.load(args.test_labels_right_file)

    mdl = model((X_train.shape[1],))
    train(mdl, X_train, y_train, X_test, y_test, args.checkpoint)
    evaluate(mdl, X_test, y_left, y_right, args.predictions)





