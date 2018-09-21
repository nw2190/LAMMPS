import numpy as np
import tensorflow as tf

from parameters import n_channels_out
from convolution_layers import *

# Specify Intermediate Channel Sizes and Node Counts
#nodes = [16, 32, 16*16*resolution]
middle_channels = 64
middle_res = 8
nodes = [10, 10, 25, 50, middle_res*middle_res*middle_channels]
#channels = [64, 32, 16, n_channels_out]
channels = [128, 64, 32, n_channels_out]

# Specify Kernel/Filter Sizes
#kernels = [3, 3, 3]
kernels = [4, 4, 4]

# Convolutional Neural Network Model
def conv_net(X):
    training = True
    
    n_ind = 0; node_count = nodes[n_ind]
    Y = dense_layer(X, node_count, training=training)

    n_ind += 1; node_count = nodes[n_ind]
    Y = dense_layer(Y, node_count, training=training)

    n_ind += 1; node_count = nodes[n_ind]
    Y = dense_layer(Y, node_count, training=training)

    n_ind += 1; node_count = nodes[n_ind]
    Y = dense_layer(Y, node_count, training=training)

    n_ind += 1; node_count = nodes[n_ind]
    Y = dense_layer(Y, node_count, training=training)

    # Reshape:  [None, 8, 8, C]
    Y = tf.expand_dims(Y,2)
    Y = tf.expand_dims(Y,3)
    Y = tf.reshape(Y, [-1, middle_res, middle_res, middle_channels])

    # [8, 8]  -->  [16, 16]
    base_res = 8
    c_ind = 0; channel_count = channels[c_ind]
    k_ind = 0; kernel_size = kernels[k_ind]
    
    Y = inception_v3(Y, channel_count, stride=1, training=training)
    Y = inception_v3(Y, channel_count, stride=1, training=training)
    
    Y = upsample(Y, base_res*2)
    
    # [16, 16]  -->  [32, 32]
    c_ind += 1; channel_count = channels[c_ind]
    k_ind += 1; kernel_size = kernels[k_ind]
    #Y = inception_v3(Y, channel_count, stride=1, training=training)
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=1, training=training)
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=1, training=training)
    
    Y = upsample(Y, base_res*4)
    
    
    # [32, 32]  -->  [64, 64]
    channel_count = n_channels_out
    k_ind += 1; kernel_size = kernels[k_ind]
    #Y = inception_v3(Y, channel_count, stride=1, training=training, activation=None)
    #Y = inception_v3(Y, channel_count, stride=1, training=training, omit_activation=True)
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=2, activation=None, add_bias=True, regularize=False, drop_rate=0.0, batch_norm=False,training=training)
    
    #Y = upsample(Y, base_res*8)
    
    if n_channels_out == 1:
        Y = tf.reshape(Y, [-1,base_res*8,base_res*8,1])
        
    logits = Y
    Y = tf.nn.sigmoid(Y)

    return Y, logits
