from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.initializers import Orthogonal, Constant
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import glob
import argparse
import json
import numpy as np 


def evaluation_report(data_dir, checkpoint, report_file, batch_size, input_shape, extract_features=False, feature_file=None, roc_file=None):
  nb_test_samples = np.sum([len(glob.glob(data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(data_dir)])
  test_steps = nb_test_samples // batch_size

  model = load_model(checkpoint)
  
  image_width, image_height = input_shape  

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_generator = test_datagen.flow_from_directory(
      data_dir,
      target_size=(image_width, image_height),
      batch_size=batch_size,
      class_mode='binary',
      shuffle=False)
  
  predictions = model.predict_generator(test_generator, steps=test_steps)
  predictions = predictions.ravel()

  y_pred = predictions > 0.5
  true_labels = test_generator.classes[:len(y_pred)]

  fpr, tpr, thresholds = roc_curve(true_labels, predictions)
  roc_auc = auc(fpr, tpr)

  #np.savetxt('fpr_stage2.txt', fpr)
  #np.savetxt('tpr_stage2.txt', tpr)

  cm = confusion_matrix(true_labels, y_pred)
  tn = cm[0][0]
  fp = cm[0][1]
  fn = cm[1][0]
  tp = cm[1][1]

  report = {}
  report['Accuracy'] = float(tp+tn)/float(tp+tn+fp+fn) 
  report['Sensitivity'] = float(tp)/float(tp+fn)
  report['Specificity'] = float(tn)/float(tn+fp)
  report['AUC'] = roc_auc
  #report['fpr'] = fpr
  #report['tpr'] = tpr
  with open(report_file, 'w') as fp:
    json.dump(report, fp)

  print('Model report. Report saved to: %s'%(report_file))
  print('Accuracy: %.2f'%report['Accuracy'])
  print('Sensitiviy: %.2f'%report['Sensitivity'])
  print('Specificity: %.2f'%report['Specificity'])
  print('AUC: %.2f'%report['AUC'])

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
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ('+roc_title+')')
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


