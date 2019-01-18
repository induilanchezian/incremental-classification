import numpy as np
from tqdm import tqdm
import sys
import argparse
import os
import glob

# use non standard flow_from_directory
from image_processing_v1 import ImageDataGenerator
# it outputs not only x_batch and y_batch but also image names

from keras.models import Model, load_model
from keras.layers import Dense

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
            type=str,
            help='Data directory containing image files'
            )
    parser.add_argument('--logits_file',
            type=str,
            help='File to store the logits dictionary')
    parser.add_argument('--prev_model',
            type=str,
            help='File path to saved model')
    
    args = parser.parse_args()
    
    datagen = ImageDataGenerator(
        data_format='channels_last',
        rescale=1. / 255
    )

    generator = datagen.flow_from_directory(
        args.data_dir,
        target_size=(512, 512),
        batch_size=64, shuffle=False,
        class_mode='binary'
    )


    nb_samples = np.sum([len(glob.glob(args.data_dir+'/'+i+'/*.jpeg')) for i in os.listdir(args.data_dir)])
    batch_size = 64
    num_batches = nb_samples // batch_size
    model = load_model(args.prev_model)
    dense_weights = model.layers[-1].get_weights()
    model.pop()
    features = model.layers[-1].output
    dense_output = Dense(1,weights=dense_weights)(features)
    model = Model(model.input, dense_output)

    batches = 0
    logits = {}

    for x_batch, _, name_batch in tqdm(generator):
        batch_logits = model.predict_on_batch(x_batch)

        for i, n in enumerate(name_batch):
            logits[n] = batch_logits[i]

        batches += 1
        if batches >= (num_batches+1):
            break

    np.save(args.logits_file, logits)
