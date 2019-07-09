import tensorflow as tf

def maxout(inputs, num_units):
    return tf.contrib.layers.maxout(inputs, num_units)

def maxout_shape(input_shape, num_units):
    output_shape = list(input_shape)
    output_shape[-1] = num_units
    return output_shape

