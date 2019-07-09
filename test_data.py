import os
import shutil 

old_dir = 'data/preprocessed_5c_original_dist'
new_dir = 'data/preprocessed_train'
subdirs = os.listdir(old_dir)
for d in subdirs:
    subd = os.path.join(old_dir, d)
    classes = os.listdir(subd)
    for d_class in classes:
        src_dir = os.path.join(subd, d_class)
        files_list = os.listdir(src_dir)
        for fl in files_list:
            f = os.path.join(src_dir, fl)
            f_cpy = os.path.join(new_dir, fl)
            shutil.copy(f, f_cpy)
