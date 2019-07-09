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

def evaluation_report(predictions_file_list, true_labels, report_file, roc_file=None):
  #classes=np.unique(true_labels)
  binary_pred1 = np.load(predictions_file_list[0])
  binary_pred1 = binary_pred1.astype(int)
  
  binary_pred2 = np.load(predictions_file_list[1])
  binary_pred2 = binary_pred2.astype(int)
  
  binary_pred3 = np.load(predictions_file_list[2])
  binary_pred3 = binary_pred3.astype(int)

  binary_pred4 = np.load(predictions_file_list[3])
  binary_pred4 = binary_pred4.astype(int)

  binary_pred5 = np.load(predictions_file_list[4])
  binary_pred5 = binary_pred5.astype(int)

  binary_pred = np.vstack((binary_pred1, binary_pred2, binary_pred3, binary_pred4, binary_pred5))
  binary_preds = stats.mode(binary_pred, axis=0) 
  binary_preds = binary_preds[0]
  binary_preds = binary_preds.ravel()
  print(binary_preds.shape)
  binary_preds = binary_preds.astype(int)

  true_labels_binary = true_labels.copy()
  true_labels_binary[true_labels_binary != 0] = 1
  print('Binary accuracy: {}'.format(np.sum(true_labels_binary == binary_preds)/len(binary_preds)))

  predictions = np.zeros((len(binary_preds), len(predictions_file_list)-5))
  for j, c in enumerate(predictions_file_list[5:]):
    pred = np.load(c)
    predictions[:,j] = pred

  y_pred = stats.mode(predictions, axis=1)
  y_pred = y_pred[0]
  y_pred = y_pred.ravel()
  y_pred = y_pred+1

  print(np.sum(binary_preds))
  print(np.sum(true_labels))
  #binary_pred[binary_pred_2 == 1] = 1 
  y_pred[binary_preds == 0] = 0

  y_pred = y_pred.astype(int)
  true_labels = true_labels.astype(int)

  #y_pred = y_pred[true_labels != 1]
  #binary_pred_2 = binary_pred_2[true_labels != 1]
  #true_labels = true_labels[true_labels != 1]

  #y_pred[y_pred == 2] = 1
  #y_pred[y_pred == 3] = 2
  #y_pred[y_pred == 4] = 3

  #true_labels[true_labels == 2] = 1
  #true_labels[true_labels == 3] = 2
  #true_labels[true_labels == 4] = 3

  classes=np.unique(true_labels)
  print(np.unique(y_pred))
  print(np.unique(true_labels))
  
  acc = accuracy_score(true_labels, y_pred)
  kappa_score = cohen_kappa_score(true_labels, y_pred, weights="quadratic")
  cnf = confusion_matrix(true_labels, y_pred)
  plt.figure()
  plot_confusion_matrix(cnf, classes=classes, normalize=False,
                      title='Unnormalized confusion matrix')
  plt.show()

  y_pred = label_binarize(y_pred, classes=classes)
  true_labels = label_binarize(true_labels, classes=classes)

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
  report['Accuracy'] = acc
  report['AUC1'] = roc_auc[0]
  report['AUC2'] = roc_auc[1]
  report['AUC3'] = roc_auc[2]
  report['AUC4'] = roc_auc[3]
  report['AUC5'] = roc_auc[4]
  report['AUC_Macro'] = roc_auc["macro"]
  report['AUC_Micro'] = roc_auc["micro"]
  report['kappa'] = kappa_score
  with open(report_file, 'w') as fp:
    json.dump(report, fp)

  print('Model report. Report saved to: %s'%(report_file))
  print('Accuracy: %.2f'%report['Accuracy'])
  print('AUC1: %.2f'%report['AUC1'])
  print('AUC2: %.2f'%report['AUC2'])
  print('AUC3: %.2f'%report['AUC3'])
  print('AUC4: %.2f'%report['AUC4'])
  print('AUC5: %.2f'%report['AUC5'])
  print('AUC (micro): %.2f'%report['AUC_Micro'])
  print('AUC (macro): %.2f'%report['AUC_Macro'])
  print('Kappa score: %.2f'%report['kappa'])

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
  parser.add_argument('--preds_files_list', type=str, default=None, help='.json file where report is saved')
  parser.add_argument('--output_file', type=str, default=None, help='.json file where report is saved')
  parser.add_argument('--roc_file', type=str, default=None, help='Image file where ROC plot is saved')

  args = parser.parse_args()

  left = np.load('left_test.npy')
  right = np.load('right_test.npy')
  #labels = np.maximum(left, right)
  labels = np.append(left, right)
  print(labels.shape)

  preds_file_list = args.preds_files_list.split(',')
  print(preds_file_list)

  args = parser.parse_args()
  evaluation_report(preds_file_list, labels, args.output_file, args.roc_file)
