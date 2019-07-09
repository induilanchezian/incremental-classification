import os
import shutil 
import numpy as np 

dir_path = '/home/indu/Thesis/incremental-classification/data/normalvsmod'

oversample_classes = ['1']
#undersample_classes = ['0']

for cls in oversample_classes:
    sub_path = os.path.join(dir_path, cls)
    filenames = os.listdir(sub_path)
    #num_choices = 1600
    old_filenames = list(map(lambda x: os.path.join(sub_path, x), filenames))
    print(old_filenames)
    #old_filenames = np.random.choice(old_filenames, num_choices)
    for i in np.arange(4):
        new_filenames = list(map(lambda x: os.path.join(sub_path, str(i)+'_'+x), filenames))
        for (f, f_cpy) in zip(old_filenames, new_filenames):
            shutil.copy(f, f_cpy)

for cls in undersample_classes:
    sub_path = os.path.join(dir_path, cls)
    filenames = os.listdir(sub_path)
    filenames = np.random.choice(filenames, 10646, replace=False)
    del_filenames = list(map(lambda x: os.path.join(sub_path,x), filenames))
    for i in del_filenames:
        try:
            os.remove(i)
        except OSError:
            print('Cannot delete file {}'.format(i))


