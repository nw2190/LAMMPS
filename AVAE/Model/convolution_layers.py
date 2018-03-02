from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import layers


ACTIVATION = tf.nn.relu
BIAS_SHIFT = 0.01

WT_REG = tf.contrib.layers.l1_regularizer(0.25)
BI_REG = tf.contrib.layers.l1_regularizer(0.25)
#WT_REG = tf.contrib.layers.l2_regularizer(10.0)
#BI_REG = tf.contrib.layers.l2_regularizer(10.0)


# Define Batch Normalization Layer
def batch_norm_layer(x,training,name=None,reuse=None):
    y = layers.batch_normalization(x,
                                   axis=-1,
                                   momentum=0.99,
                                   epsilon=0.001,
                                   center=True,
                                   scale=True,
                                   beta_initializer=tf.zeros_initializer(),
                                   gamma_initializer=tf.ones_initializer(),
                                   moving_mean_initializer=tf.zeros_initializer(),
                                   moving_variance_initializer=tf.ones_initializer(),
                                   beta_regularizer=None,
                                   gamma_regularizer=None,
                                   training=training,
                                   trainable=True,
                                   name=name,
                                   reuse=reuse,
                                   renorm=False,
                                   renorm_clipping=None,
                                   renorm_momentum=0.99)
    return y


# Define Convolutional Layer
def conv2d_layer(x, n_out, kernel_size, stride=1, activation=ACTIVATION, regularize=True, drop_rate=0.0, batch_norm=False, training=True, name=None, reuse=None):

    if batch_norm:
        if name:
            x = batch_norm_layer(x,training,name=name + '_bn', reuse=reuse)
        else:
            x = batch_norm_layer(x,training,name=name, reuse=reuse)
    
    wt_init = tf.truncated_normal_initializer(stddev=0.1)
    bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.01)

    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None
        
    y = layers.conv2d(x,
                      n_out,
                      kernel_size,
                      strides=(stride,stride),
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=(1,1),
                      activation=activation,
                      use_bias=True,
                      kernel_initializer=wt_init,
                      bias_initializer=bi_init,
                      kernel_regularizer=wt_reg,
                      bias_regularizer=bi_reg,
                      activity_regularizer=None,
                      trainable=True,
                      name=name,
                      reuse=reuse)
    #y = layers.dropout(y, rate=drop_rate, training=training)
    return y





# Define Convolution Transpose Layer
def transpose_conv2d_layer(x, n_out, kernel_size, stride=1, activation=ACTIVATION, add_bias=True, regularize=True, drop_rate=0.0, batch_norm=False,training=True, name=None, reuse=None):

    if batch_norm:
        if name:
            x = batch_norm_layer(x,training,name=name + '_bn', reuse=reuse)
        else:
            x = batch_norm_layer(x,training,name=name, reuse=reuse)
            
    wt_init = tf.truncated_normal_initializer(stddev=0.1)
    bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.01)

    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None
        

    y = layers.conv2d_transpose(x,
                                n_out,
                                kernel_size=[kernel_size,kernel_size],
                                strides=(stride, stride),
                                padding='same',
                                data_format='channels_last',
                                activation=activation,
                                use_bias=add_bias,
                                kernel_initializer=wt_init,
                                bias_initializer=bi_init,
                                kernel_regularizer=wt_reg,
                                bias_regularizer=bi_reg,
                                activity_regularizer=None,
                                trainable=True,
                                name=name,
                                reuse=reuse)
    
    #y = layers.dropout(y, rate=drop_rate, training=training)
    return y



# Define Fully Connected Layer
def dense_layer(x, n_out, activation=ACTIVATION, drop_rate=0.0, reuse=None, name=None, batch_norm=False, regularize=True, training=True):

    if batch_norm:
        if name:
            x = batch_norm_layer(x,training,name=name + '_bn', reuse=reuse)
        else:
            x = batch_norm_layer(x,training,name=name, reuse=reuse)

    #wt_init = tf.truncated_normal_initializer(stddev=0.15)
    #bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.25)
    wt_init = tf.truncated_normal_initializer(stddev=0.05)
    bi_init = tf.truncated_normal_initializer(mean=BIAS_SHIFT,stddev=0.35)
    
    if regularize:
        wt_reg = WT_REG
        bi_reg = BI_REG
    else:
        wt_reg = None
        bi_reg = None

    y = layers.dense(x,
                     n_out,
                     activation=activation,
                     use_bias=True,
                     kernel_initializer=wt_init,
                     bias_initializer=bi_init,
                     kernel_regularizer=wt_reg,
                     bias_regularizer=bi_reg,
                     trainable=True,
                     name=name,
                     reuse=reuse)
    
    #y = layers.dropout(y, rate=drop_rate, training=training)
    return y



# Defines Inception V3 Layer
# http://arxiv.org/abs/1512.00567
def inception_v3(x, n_out, stride=1, activation=ACTIVATION, regularize=True, drop_rate=0.0, batch_norm=False, training=True, name=None, reuse=None):

    # Store name to use as prefix
    base_name = name

    ###############################
    """  1x1 CONV  +  3x3 CONV  """
    ###############################
    if name:  name = base_name + '_1a'
    y1 = conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y1 = conv2d_layer(y1, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ############################################
    """  1x1 CONV  +  3x3 CONV  +  3x3 CONV  """
    ############################################
    
    if name:  name = base_name + '_2a'
    y2 = conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2b'
    y2 = conv2d_layer(y2, n_out//4, 3, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2c'
    y2 = conv2d_layer(y2, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ###################################
    """  3x3 MAX POOL  +  1x1 CONV  """
    ###################################

    y3 = layers.max_pooling2d(x, 3, stride, padding='same', data_format='channels_last')

    if name:  name = base_name + '_3'
    y3 = conv2d_layer(y3, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ##################
    """  1x1 CONV  """
    ##################

    if name:  name = base_name + '_4'
    y4 = conv2d_layer(x, n_out//4, 1, stride=stride, activation=activation, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = tf.concat([y1,y2,y3,y4],3)

    return y




# Defines Inception V3 Layer Transpose
# http://arxiv.org/abs/1512.00567
def transpose_inception_v3(x, n_out, stride=1, activation=ACTIVATION, regularize=True, drop_rate=0.0, batch_norm=False, training=True, name=None, reuse=None):

    # Store name to use as prefix
    base_name = name

    ###############################
    """  1x1 CONV  +  3x3 CONV  """
    ###############################
    if name:  name = base_name + '_1a'
    y1 = transpose_conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y1 = transpose_conv2d_layer(y1, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ############################################
    """  1x1 CONV  +  3x3 CONV  +  3x3 CONV  """
    ############################################
    
    if name:  name = base_name + '_2a'
    y2 = transpose_conv2d_layer(x, n_out//4, 1, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2b'
    y2 = transpose_conv2d_layer(y2, n_out//4, 3, stride=1, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_2c'
    y2 = transpose_conv2d_layer(y2, n_out//4, 3, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ###################################
    """  3x3 MAX POOL  +  1x1 CONV  """
    ###################################

    y3 = layers.max_pooling2d(x, 3, 1, padding='same', data_format='channels_last')

    if name:  name = base_name + '_3'
    y3 = transpose_conv2d_layer(y3, n_out//4, 1, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    ##################
    """  1x1 CONV  """
    ##################

    if name:  name = base_name + '_4'
    y4 = transpose_conv2d_layer(x, n_out//4, 1, stride=stride, activation=activation, regularize=regularize,
                                drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = tf.concat([y1,y2,y3,y4],3)

    return y





# Defines ResNet Layer
def resnet(x, n_out, kernel_size, activation=ACTIVATION, regularize=True, drop_rate=0.0, batch_norm=False, training=True, name=None, reuse=None):

    # Store name to use as prefix
    base_name = name

    if name:  name = base_name + '_1a'
    y = conv2d_layer(x, n_out, kernel_size, stride=1, activation=activation, regularize=regularize,
                     drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    if name:  name = base_name + '_1b'
    y = conv2d_layer(y, n_out, kernel_size, stride=1, activation=None, regularize=regularize,
                      drop_rate=drop_rate, batch_norm=batch_norm, training=training, name=name, reuse=reuse)

    y = activation(tf.add(x,y))
    return y
