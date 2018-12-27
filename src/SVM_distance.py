import os
import sys

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import hinge_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def create_dataframe(directory, label):
	files_list = os.listdir(directory)
	files_rnd = np.random.choice(files_list, 500)

	#Read features for the first file and store in a data frame
	df = pd.read_csv(directory+files_rnd[0], sep=",", header=None)
	for f in files_rnd[1:]:
		df_next = pd.read_csv(directory+f, sep=",", header=None)
		df = pd.concat([df,df_next])
	df['label'] = label
	return df

if __name__=='__main__':
	classes = ['0', '1']

	#directory = '/tmp/bottleneck/'+str(0)+'/'
	#df = create_dataframe(directory, 0)
        features = []
	for cls in classes:
		fname = 'base_set_features_class_'+cls+'_stage1.txt'
		#directory = '/tmp/bottleneck/'+str(cls)+'/'
		#df_next = create_dataframe(directory, cls)
		#df = pd.concat([df, df_next])
		features.append(np.loadtxt(fname))
	print(features[0].shape)
	print(features[1].shape)
	labels = np.zeros(features[0].shape[0])
	labels = np.append(labels, np.ones(features[1].shape[0]))
	features = np.concatenate(features)
	print(features.shape)

	#svc = LinearSVC(random_state=0, tol=1e-5)
	svc = SVC(random_state=0, kernel='linear')
	#initial_preds = svc.predict(features)
	#initial_dist = hinge_loss(labels, initial_preds)
	#print('Initial margin (approximate): %.2f'%(initial_dist))

	svc.fit(features, labels)

	mode = 'linear'
	if (mode == 'linear'):
		a = svc.coef_[0]
		norm_a = np.linalg.norm(a, ord=2)
		dist = 2.0/norm_a
		print('Distance between hyperplanes: %.2f'%(dist))

		distances = svc.decision_function(features)
		pos_distances = np.min(distances[np.where(distances>0)])
		neg_distances = np.min(distances[np.where(distances<0)])
		print('Minimum signed distance (for positive class): %.2f'%(pos_distances))
		print('Minimum signed distance (for negative class): %.2f'%(neg_distances))
	
		print('Total loss: %.2f'%(dist+pos_distances-neg_distances))

		print(svc.intercept_)
	
	final_preds = svc.predict(features)
	np.savetxt('labels.txt',labels)
	np.savetxt('preds.txt', final_preds)
	final_dist = hinge_loss(labels, final_preds)
	print('Margin (approximate): %.2f'%(final_dist))
	print('Number of misclassifications: %d'%(len(labels) - accuracy_score(labels,final_preds,normalize=False)))
	print('Accuracy: %f'%accuracy_score(labels, final_preds))
	print('Confusion Matrix:')
	print(confusion_matrix(labels, final_preds))
