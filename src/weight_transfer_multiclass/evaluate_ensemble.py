from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
import keras.backend as K
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.preprocessing import label_binarize
from scipy import interp, stats
import os
import glob
import argparse
import json
import numpy as np 
import itertools

from model1 import model as m1
from model2 import model as m2

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

def predict(test_data_dir, checkpoint, shp, batch_size):
  nb_test_samples = np.sum([len(glob.glob(test_data_dir+'/'+i+'/*.jpeg')) for i in ['1','2','3','4']])
  lambda_const=0.1
  test_steps = nb_test_samples // batch_size

  if K.image_data_format() == 'channels_first':
    input_shape = (3, shp[0], shp[1])
  else:
    input_shape = (shp[0], shp[1], 3)

  if 'smaller' in checkpoint:
      model = m2(input_shape)
      model.load_weights(checkpoint)
  else:
      model = m1(input_shape)
      model.load_weights(checkpoint)
  image_width, image_height = shp 

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(
      test_data_dir,
      target_size=(image_width, image_height),
      batch_size=batch_size,
      class_mode='sparse',
      classes=['1','2','3','4'],
      shuffle=False)
  
  predictions = model.predict_generator(test_generator, steps=test_steps)
  predictions = predictions.ravel()

  y_pred = np.clip(predictions, 0.0, 4.0)
  y_pred[y_pred<=0.5] = 0
  y_pred[np.logical_and(y_pred>0.5, y_pred <=1.5)] = 1
  y_pred[np.logical_and(y_pred>1.5, y_pred <=2.5)] = 2
  y_pred[y_pred>=2.5] = 3
  
  true_labels = test_generator.classes[:len(y_pred)]
  classes = np.unique(true_labels)

  del model
  return y_pred, true_labels, classes


def evaluation_report(test_data_dir, checkpoints_list, report_file, batch_size, input_shape, extract_features=False, feature_file=None, roc_file=None):
  pred, true_labels, classes = predict(test_data_dir, checkpoints_list[0], input_shape, batch_size)
  predictions = np.zeros((len(pred), len(checkpoints_list)))
  predictions[:,0] = pred
  for j, c in enumerate(checkpoints_list[1:]):
    pred, true_labels, classes = predict(test_data_dir, c, input_shape, batch_size)
    predictions[:,j+1] = pred

  print(predictions.shape)
  y_pred = stats.mode(predictions, axis=1)
  y_pred = y_pred[0]
  print(y_pred.shape)
  y_pred = y_pred.ravel()

  y_pred = y_pred.astype(int)
  true_labels = true_labels.astype(int)
  
  acc = accuracy_score(true_labels, y_pred)
  kappa_score = cohen_kappa_score(true_labels, y_pred, weights="quadratic")
  cnf = confusion_matrix(true_labels, y_pred)
  plt.figure()
  plot_confusion_matrix(cnf, classes=classes, normalize=True,
                      title='Normalized confusion matrix')
  plt.show()

  plt.figure()
  plot_confusion_matrix(cnf, classes=classes, normalize=False,
                      title='Confusion matrix')
  plt.show()

  y_pred = label_binarize(y_pred, classes=classes)
  true_labels = label_binarize(true_labels, classes=classes)

  print(true_labels.shape)
  print(y_pred.shape)

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
  #np.savetxt('fpr_stage2.txt', fpr)
  #np.savetxt('tpr_stage2.txt', tpr)
  
  report = {}
  report['Accuracy'] = acc
  report['AUC1'] = roc_auc[0]
  report['AUC2'] = roc_auc[1]
  report['AUC3'] = roc_auc[2]
  report['AUC4'] = roc_auc[3]
  report['AUC_Macro'] = roc_auc["macro"]
  report['AUC_Micro'] = roc_auc["micro"]
  report['kappa'] = kappa_score
  #report['fpr'] = fpr
  #report['tpr'] = tpr
  with open(report_file, 'w') as fp:
    json.dump(report, fp)

  print('Model report. Report saved to: %s'%(report_file))
  print('Accuracy: %.2f'%report['Accuracy'])
  print('AUC1: %.2f'%report['AUC1'])
  print('AUC2: %.2f'%report['AUC2'])
  print('AUC3: %.2f'%report['AUC3'])
  print('AUC4: %.2f'%report['AUC4'])
  print('AUC (micro): %.2f'%report['AUC_Micro'])
  print('AUC (macro): %.2f'%report['AUC_Macro'])
  print('Kappa score: %.2f'%report['kappa'])

  if extract_features:
    features_model = Model(model.inputs, model.layers[-2].output)
    features = features_model.predict_generator(
      test_generator,
      steps=test_steps)

    assert feature_file is not None 
    np.savetxt(feature_file, features)

  if roc_file is not None:
    roc_title = roc_file.split('.')
    roc_title = roc_title[0]
    plt.figure()
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic \n ('+roc_title+')')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(roc_file)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, help='Data directory on which to evaluate model')
  parser.add_argument('--model', type=str, help='Path to the checkpoint file of model to be evaluated')
  parser.add_argument('--output_file', type=str, help='Path to the output file which stores the reports')
  parser.add_argument('--batch_size', type=int, help='Batch size')
  parser.add_argument('--image_width', type=int, help='Image width')
  parser.add_argument('--image_height', type=int, help='Image height')
  parser.add_argument('--extract_features', type=bool, default=False, help='Boolean value indicating whether features is to be stored')
  parser.add_argument('--feature_file', type=str, default=None, help='File where features are stored')
  parser.add_argument('--roc_file', type=str, default=None, help='Image file where ROC plot is saved')

  args = parser.parse_args()
  ckpt_list = args.model.split(',')
  evaluation_report(args.data_dir, ckpt_list, args.output_file, args.batch_size, (args.image_width, args.image_height),
                    args.extract_features, args.feature_file, args.roc_file)


