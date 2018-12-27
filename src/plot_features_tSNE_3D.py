import os
import sys

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D 
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
	classes = np.arange(5)

	directory = '/tmp/bottleneck/'+str(0)+'/'
	df = create_dataframe(directory, 0)
	
	for cls in classes[1:]:
		directory = '/tmp/bottleneck/'+str(cls)+'/'
		df_next = create_dataframe(directory, cls)
		df = pd.concat([df, df_next])
	
	labels = df['label'].values
	features = df.loc[:, df.columns != 'label'].values

	tsne = TSNE(n_components=3, learning_rate=50.0,random_state=0)
	features_new = tsne.fit_transform(features)
	print features_new.shape

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	colors = ['b', 'g', 'r', 'y', 'c']
	markers = ["o", "^", "s", "P", "*"]
	lbls = ['0', '1', '2', '3','4','5']
	for cls, c, m,l in zip(classes, colors, markers,lbls):
		ind = np.where(labels == cls)
		ind = ind[0]
		print features_new[ind,:].shape
		ax.scatter(features_new[ind,0], features_new[ind,1], features_new[ind,2],c=c, label=l,marker=m)
	ax.legend()
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	plt.show()
