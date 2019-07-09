import numpy as np 

def cluster_radius(a):
    mean_a = np.mean(a, axis=0)
    dists = np.sqrt(np.sum((a[:,None] - mean_a[None,:])**2, axis=-1))
    print(dists.shape)
    radius = np.mean(dists)
    return radius 
    
def mean_distances(a, b):
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    dist = np.sqrt(np.sum((mean_a-mean_b)**2))
    return dist

mild_feats = np.load('modelA1nl_100_data1_1.npy')
mod_feats = np.load('modelA1nl_100_data1_2.npy')
sev_feats = np.load('modelA1nl_100_data1_3.npy')
pro_feats = np.load('modelA1nl_100_data1_4.npy')

feats = [mild_feats, mod_feats, sev_feats, pro_feats]
classes=['MILD', 'MODERATE','SEVERE', 'PROLIFERATIVE']

for i in np.arange(len(feats)):
    for j in np.arange(i+1):
        dist_mean = mean_distances(feats[i], feats[j])
        if (i == j):
            continue
        else:
            print('Inter cluster mean distance for classes {} and {}: {}'.format(classes[i], classes[j], dist_mean))


for i in np.arange(len(feats)):
    radius = cluster_radius(feats[i])
    print('Cluster radius for class {}: {}'.format(i, radius))

