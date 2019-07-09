from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from base_train_multiclass_model2 import accuracy, maxout, maxout_shape
from scipy import interp
import os
import glob
import argparse
import json
import numpy as np 
import itertools

def quadratic_kappa(y, t, eps=1e-15):
      # Assuming y and t are one-hot encoded!
      num_scored_items = y.shape[0]
      num_ratings = y.shape[1] 
      ratings_mat = np.tile(np.arange(0, num_ratings)[:, None], 
                            reps=(1, num_ratings))
      ratings_squared = (ratings_mat - ratings_mat.T) ** 2
      weights = ratings_squared / (float(num_ratings) - 1) ** 2
	
      # We norm for consistency with other variations.
      y_norm = y / (eps + y.sum(axis=1)[:, None])
	
      # The histograms of the raters.
      hist_rater_a = y_norm.sum(axis=0)
      hist_rater_b = t.sum(axis=0)
	
      # The confusion matrix.
      conf_mat = np.dot(y_norm.T, t)
	
      # The nominator.
      nom = np.sum(weights * conf_mat)
      expected_probs = np.dot(hist_rater_a[:, None], 
                              hist_rater_b[None, :])
      # The denominator.
      denom = np.sum(weights * expected_probs / num_scored_items)
	
      return 1 - nom / denom

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

def evaluation_report(data_dir, checkpoint, report_file, batch_size, input_shape, extract_features=False, feature_file=None, roc_file=None):
  nb_test_samples = np.sum([len(glob.glob(data_dir+'/'+i+'/*.jpeg')) for i in ['0','1','2','3','4']])
  test_steps = nb_test_samples // batch_size
  
  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(
      data_dir,
      target_size=(image_width, image_height),
      batch_size=batch_size,
      class_mode='sparse',
      classes=['0','1','2','3','4'],
      shuffle=False)

  model_stage_0 = load_model('../../models/base/modelv1-KaggleDRD-50ep-base.h5')
  predictions_stage_0 = model_stage_0.predict_generator(test_generator, steps=test_steps)
  predictions_stage_0 = predictions_stage_0.ravel()

  del model_stage_0

  model = load_model(checkpoint, custom_objects={'accuracy': accuracy, 'maxout': maxout, 'maxout_shape': maxout_shape})
  
  image_width, image_height = input_shape  

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(
      data_dir,
      target_size=(image_width, image_height),
      batch_size=batch_size,
      class_mode='sparse',
      classes=['0','1','2','3','4'],
      shuffle=False)

  predictions = model.predict_generator(test_generator, steps=test_steps)
  predictions = predictions.ravel()

  y_pred = np.clip(predictions, 0.0, 4.0)
  print(y_pred[[1,5, 885,900, 2401, 2409, 2700, 2724]])
  y_pred[y_pred<=0.5] = 0
  y_pred[np.logical_and(y_pred>0.5, y_pred <=1.5)] = 1
  y_pred[np.logical_and(y_pred>1.5, y_pred <=2.5)] = 2
  y_pred[y_pred>=2.5] = 3
  print(y_pred[[1,5, 885,900, 2401, 2409, 2700, 2724]])
  true_labels = test_generator.classes[:len(y_pred)]
  print(true_labels[[1,5, 885,900, 2401, 2409, 2700, 2724]])
  classes = np.unique(true_labels)

  y_pred = y_pred.astype(int)
  predictions_stage_0 = predictions_stage_0.astype(int)
  predictions_stage_0[predictions_stage_0 > 0] = y_pred
  y_pred = predictions_stage_0
  true_labels = true_labels.astype(int)
  
  cnf = confusion_matrix(true_labels, y_pred)
  plt.figure()
  plot_confusion_matrix(cnf, classes=classes, normalize=True,
                      title='Normalized confusion matrix')
  plt.show()

  y_pred = label_binarize(y_pred, classes=classes)
  true_labels = label_binarize(true_labels, classes=classes)
  kappa_score = quadratic_kappa(y_pred, true_labels)

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

  cm = confusion_matrix(true_labels[:,1], y_pred[:,1])
  tn = cm[0][0]
  fp = cm[0][1]
  fn = cm[1][0]
  tp = cm[1][1]
  
  report = {}
  report['Accuracy'] = float(tp+tn)/float(tp+tn+fp+fn) 
  report['Sensitivity'] = float(tp)/float(tp+fn)
  report['Specificity'] = float(tn)/float(tn+fp)
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
  print('Sensitiviy: %.2f'%report['Sensitivity'])
  print('Specificity: %.2f'%report['Specificity'])
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
  evaluation_report(args.data_dir, args.model, args.output_file, args.batch_size, (args.image_width, args.image_height),
                    args.extract_features, args.feature_file, args.roc_file)


