import numpy as np 

def pairwise_distances(a, b):
    dist_matrix = np.sqrt(np.sum((a[:,None] - b[None, :])**2, axis=-1))
    return dist_matrix

mild_feats = np.load('modelA1nl_100_data1_1.npy')
mod_feats = np.load('modelA1nl_100_data1_2.npy')
sev_feats = np.load('modelA1nl_100_data1_3.npy')
pro_feats = np.load('modelA1nl_100_data1_4.npy')

feats = [mild_feats, mod_feats, sev_feats, pro_feats]
classes=['MILD', 'MODERATE','SEVERE', 'PROLIFERATIVE']

for i in np.arange(len(feats)):
    for j in np.arange(i+1):
        dist_mean = np.mean(pairwise_distances(feats[i], feats[j]))
        dist_std = np.std(pairwise_distances(feats[i], feats[j]))
        if (i == j):
            print('Intra cluster distance mean and std for class {}: {}, {}'.format(classes[i], dist_mean, dist_std))
        else:
            print('Inter cluster distance mean and std for classes {} and {}: {}, {}'.format(classes[i], classes[j], dist_mean, dist_std))


            
