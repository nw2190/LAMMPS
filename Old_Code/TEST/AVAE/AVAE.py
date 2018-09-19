#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
from tensorflow import layers
from convolution_layers import *

# Variational Autoencoder
class AVAE(object):
    model_name = "AVAE"

    # Initialize Model
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):

        # Specify whether or not to use onehot input encoding
        self.use_onehot = False

        # Specify whether to use clipping or sigmoid on outputs
        self.use_clipping = False
        self.use_sigmoid = True

        # Specify whether or not to use VAE structure
        self.use_VAE = False
        
        # Specify whether or not to use transformations
        self.use_transformations = True
        
        # resolution
        self.res = 64
        self.input_height = self.res
        self.input_width = self.res
        self.output_height = self.res
        self.output_width = self.res
        
        # Model parameters
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        # Learning rate
        self.learning_decay = 0.90
        self.start_rate = 0.00075
        #self.start_rate = 0.00015
        #self.start_rate = 0.0001
        #self.start_rate = 0.00005
        
        self.learn_startstep = int(40*750//45)
        #self.learn_midstep = int(40*1500//45)
        self.learn_midstep = int(40*2500//45)
        #self.learn_endstep = int(100*1500//45)
        self.learn_endstep = int(40*7500//45)
        self.learning_rate = self.start_rate

        # Validation batches
        self.validation_step = 400
        self.vbatch_size = self.batch_size

        # Early stopping criterion
        self.stopping_step = 200
        self.stopping_size = self.batch_size
        self.stopping_level = 0.009

        # Latent connections for decoder
        self.latent_res = 8
        self.latent_chans = 64

        # Loss function weights
        self.learning_decay = 0.99
        self.loss_tolerance = 0.33
        self.adv_loss_tolerance = 1.0
        self.kl_loss_tolerance = 10.0
        self.l2_weight = 1.0
        self.reg_weight = 0.00005
        self.kl_weight = 20.0
        self.adv_weight = 1.0


        ## Learning rate weights (Adversarial)
        self.d_weight = 1.0   # Discriminator
        self.g_weight = 1.0   # Generator

        ## Learning rate weights (L2/KL)
        self.g2_weight = 1.0  # Generator
        self.e_weight = 7.5   # Encoder
        self.a_weight = 2.0   # All (except discriminator)


        template = np.load('../Arrays/template.npy')
        template = np.array([template[:,:,0]])
        self.template = np.transpose(template,[1,2,0])

        
        # Specify number of plots to save
        self.plots = 1
        self.plot_step = 100

        # Specify when to display
        self.display_step = 25

        # channel count
        self.c_dim = 1
        
        # latent vector dimension
        self.z_dim = z_dim
        
        # load input data
        if self.use_onehot:
            data_X = np.load('../DATA_ONEHOT.npy')
        else:
            data_X = np.load('../DATA.npy')
        total_size = len(data_X)
        """
        input_scaling = 0.1
        self.data_X = input_scaling*data_X[:int(4*total_size//5)]
        self.vdata_X = input_scaling*data_X[int(4*total_size//5):]
        """
        self.data_X = data_X[:int(4*total_size//5)]
        self.vdata_X = data_X[int(4*total_size//5):]

        # load targets
        data_y = np.load('../TARGETS.npy')
        self.data_y = data_y[:int(4*total_size//5)]
        self.vdata_y = data_y[int(4*total_size//5):]
        
        # specify transformations
        if self.use_onehot:
            self.transformations = [[0,0]]
            #self.transformations = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]]
        else:
            self.transformations = [[0,0]]

        self.transformation_count = int(len(self.transformations))
        
        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size
        self.num_vbatches = len(self.vdata_X) // self.vbatch_size
        
    # Sample from multivariate Gaussian
    def sampleGaussian(self, mean, log_sigma):
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mean + 1.0 * epsilon * tf.exp(log_sigma)

    # Encoder
    def encoder(self, x):
        with tf.variable_scope("encoder", reuse=None):
            """
            if self.use_onehot:
                net = tf.expand_dims(x,3)
                net = conv2d_layer(net, 4, 3, stride=1, training=self.is_training, name='conv_1')
                net = conv2d_layer(net, 8, 3, stride=1, training=self.is_training, name='conv_2')
                net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
                #net = inception_v3(net, 4, stride=2, training=self.is_training, name='incept_1')
                net = conv2d_layer(net, 8, 3, stride=1, training=self.is_training, name='conv_3')
                net = conv2d_layer(net, 16, 3, stride=1, training=self.is_training, name='conv_4')
                net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
                #net = inception_v3(net, 8, stride=2, training=self.is_training, name='incept_2')
                #net = inception_v3(net, 16, stride=2, training=self.is_training, name='incept_3')
                net = conv2d_layer(net, 16, 2, stride=1, training=self.is_training, name='conv_5')
                net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last')

                net = tf.reshape(net,[-1,int(16*self.res//8*self.res//8)])
            """

            if self.use_onehot:
                # res = 32
                net = tf.expand_dims(x,3)
                #net_lay1 = layers.max_pooling2d(net, 2, 1, padding='same', data_format='channels_last')
                #net_lay2 = layers.max_pooling2d(net, 3, 1, padding='same', data_format='channels_last')
                #net = tf.concat([net,net_alt],3)
                #net = tf.add(tf.add(net,net_lay1),net_lay2)
                net_lay1 = layers.max_pooling2d(net, 3, 1, padding='same', data_format='channels_last')
                net = tf.add(net,net_lay1)
                net = conv2d_layer(net, 32, 5, stride=1, add_bias=False, training=self.is_training, name='en_conv_1')
                net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
                
                # res = 16
                net = conv2d_layer(net, 64, 3, stride=1, add_bias=False, training=self.is_training, name='en_conv_2')
                #net = inception_v3(net, 16, stride=1, add_bias=False, training=self.is_training, name='e_incept_1')
                net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
                
                # res = 8
                #net = conv2d_layer(net, 8, 2, stride=1, training=is_training, name='d_conv_3')
                net = inception_v3(net, 8, stride=1, add_bias=False, training=self.is_training, name='en_incept_2')
                net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last')

                # res = 4
                #net = tf.reshape(net,[-1,int(16*self.res//8*self.res//8)])
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(linear(net, self.z_dim, scope='en_fc1'))

            else:
                net = dense_layer(x, 16, training=self.is_training,regularize=False, name='en_dense_1')
                net = dense_layer(net, 32, training=self.is_training,regularize=False, name='en_dense_2')
                net = dense_layer(net, 8*8*64, training=self.is_training,regularize=False, name='en_dense_3')



            if self.use_VAE:
                # Dense encoding layer
                gaussian_params = dense_layer(net, 2*self.z_dim, training=self.is_training,
                                              regularize=False, activation = None, name='en_dense_enc')
            
                # Gaussian parameters
                mean = gaussian_params[:, :self.z_dim]
                logsigma = gaussian_params[:, self.z_dim:]
            else:
                mean = net
                logsigma = net
                
        return mean, logsigma


    def smooth_clip(self,x):
        x = tf.where(x > 1.0,
                     0.5 * tf.square(x) + 0.5,
                     x)
        x = tf.where(x < 0.0,
                     -0.5 * tf.square(x) + x,
                     x)
        return x
        
    
    # Decoder
    def decoder(self, z, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):

            if self.use_VAE:
                # Dense decoding layer
                net = dense_layer(z, self.latent_res*self.latent_res*self.latent_chans,
                                  regularize=False, training=self.is_training, name='g_dense_dec')
            else:
                net = z
                
            # Reshape
            net = tf.reshape(net, [-1, self.latent_res, self.latent_res, self.latent_chans])

            net = transpose_conv2d_layer(net, 32, kernel_size=3, stride=2, training=self.is_training, name='g_tconv_1', reuse=reuse)
            net = transpose_conv2d_layer(net, 16, kernel_size=3, stride=2, training=self.is_training, name='g_tconv_2', reuse=reuse)
            net = transpose_conv2d_layer(net, self.c_dim, kernel_size=3, stride=2, activation=None,
                                         add_bias=False, regularize=False, training=self.is_training, name='g_tconv_4', reuse=reuse)

            if self.use_clipping:
                out = self.smooth_clip(net)
            elif self.use_sigmoid:
                out = tf.nn.sigmoid(net)
            else:
                out = net

            """
            # Dense decoding layer
            net = dense_layer(z, self.latent_res*self.latent_res*self.latent_chans,
                              regularize=False, training=self.is_training, name='dense_dec')
            # Reshape
            net = tf.reshape(net, [-1, self.latent_res, self.latent_res, self.latent_chans])

            # Res/16
            #net = transpose_conv2d_layer(net, 96, kernel_size=2, stride=2, training=self.is_training, name='tconv_1', reuse=reuse)
            #net = transpose_inception_v3(net, 96, stride=2, training=self.is_training, name='tincept_0', reuse=reuse)

            # Res/8
            net = transpose_inception_v3(net, 32, stride=2, training=self.is_training, name='tincept_1', reuse=reuse)

            # Res/4
            #net = transpose_conv2d_layer(net, 32, kernel_size=3, stride=2, training=self.is_training, name='tconv_2', reuse=reuse)
            net = transpose_inception_v3(net, 16, stride=2, training=self.is_training, name='tincept_2', reuse=reuse)

            # Res/2
            net = transpose_conv2d_layer(net, self.c_dim, kernel_size=3, stride=2, activation=tf.nn.sigmoid,
                                         add_bias=False, regularize=False, training=self.is_training, name='tconv_3', reuse=reuse)
            out = net
            """
            return out

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            # res = 32
            net = conv2d_layer(x, 32, 3, stride=1, training=is_training, name='d_conv_1')
            net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
            
            # res = 16
            net = conv2d_layer(net, 16, 3, stride=1, training=is_training, name='d_conv_2')
            net = layers.max_pooling2d(net, 3, 2, padding='same', data_format='channels_last')
            
            # res = 8
            #net = conv2d_layer(net, 8, 2, stride=1, training=is_training, name='d_conv_3')
            net = inception_v3(net, 8, stride=1, training=is_training, name='d_incept_1')
            net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last')
            
            # res = 4
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(linear(net, 256, scope='d_fc1'))
            out_logit = linear(net, 1, scope='d_fc2')
            out = tf.nn.sigmoid(out_logit)
            
            """
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
            """
            
        return out, out_logit
    

    # Build TensorFlow Graph
    def build_model(self):
        # data dimensions
        if self.use_onehot:
            in_dims = [self.res,self.res]
        else:
            in_dims = [2]
            
        data_dims = [self.input_height, self.input_width, self.c_dim]

        bs = self.batch_size

        """ Graph Input """
        # Input/output placeholders
        self.inputs = tf.placeholder(tf.float32, [bs] + in_dims, name='inputs')
        self.outputs = tf.placeholder(tf.float32, [bs] + data_dims, name='outputs')

        # Training/reuse placeholders
        self.is_training = tf.placeholder(tf.bool, name='training')
        self.reuse = tf.placeholder(tf.bool, name='reuse')

        # Placeholders for adaptive loss function weights
        self.kl_c = tf.placeholder(tf.float32, name='kl_c')
        self.reg_c = tf.placeholder(tf.float32, name='reg_c')
        self.adv_c = tf.placeholder(tf.float32, name='adv_c')
        self.l_rate = tf.placeholder(tf.float32, name='l_rate')
        
        """ Loss Function """

        # Encode
        self.mu, self.logsigma = self.encoder(self.inputs)
        self.sigma = tf.exp(self.logsigma)
        
        # Sample in latent space
        z = self.sampleGaussian(self.mu,self.logsigma)

        """
        self.mu, self.logsigma = self.encoder(self.inputs)
        z = self.mu
        """
        
        # Decode
        out = self.decoder(z)
        self.pred = out




        # Template storing locations of atoms before impact
        template_tiled = tf.tile(tf.expand_dims(self.template, 0), [self.batch_size,1,1,1])

        # Find Interior Indices
        interior_indices = tf.not_equal(template_tiled,0.0)

        y = self.outputs
        pred = self.pred
        
        # Note: solutions are set to 0.0001 the locations of undamaged atoms
        empty_indices = tf.equal(y,0.0)
        empty_int_indices = tf.logical_and(interior_indices, empty_indices)

        zero_tensor = tf.zeros_like(y)
        eps_tensor = tf.add(zero_tensor, 0.0001)
        ones_tensor = tf.add(zero_tensor, 1.0)

        # Masked Versions
        masked_pred = tf.where(interior_indices, tf.maximum(pred,eps_tensor), zero_tensor)
        masked_pred = tf.minimum(masked_pred,ones_tensor)
        
        # Remove 'displaced' atoms from solution
        masked_y = tf.where(interior_indices, y, zero_tensor)
        
        # Declare 'displaced' atoms to be fully damaged
        masked_y = tf.where(empty_int_indices, ones_tensor, masked_y)



        self.outputs = masked_y
        self.pred = masked_pred
        
        # Mean Square Cost Function
        #self.L2_loss = tf.reduce_mean(tf.reduce_mean(tf.pow(self.pred-self.outputs, 2), axis=[1,2]))
        self.L2_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(self.pred-self.outputs, 2), axis=[1,2,3]))
        #self.L2_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(self.pred-self.outputs), axis=[1,2,3]))
            
        """    
        # Regularization loss
        self.reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.Reg_loss = sum(self.reg_list)

        """
        # KL loss
        self.KL_divergence = -0.5 * tf.reduce_mean(1 + 2 * self.logsigma - self.mu**2 - tf.exp(2 * self.logsigma), axis=1)
        self.KL_divergence = tf.reduce_mean(self.KL_divergence)

        # Total loss
        #self.loss = self.l2_weight*self.L2_loss + self.kl_c*self.KL_divergence + self.reg_c*self.Reg_loss

        if self.use_VAE:
            self.loss = self.l2_weight*self.L2_loss + self.kl_c*self.KL_divergence
        else:
            self.loss = self.l2_weight*self.L2_loss

        self.Reg_loss = self.L2_loss



        """ Adversarial Loss Function """
        # output of D for real data
        D_real, D_real_logits = self.discriminator(self.outputs)

        # output of D for fake data
        D_fake, D_fake_logits = self.discriminator(self.pred, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.Adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        self.g_loss = self.Adv_loss


        
        
        """ Training """
        """
        # Optimizer
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-06).minimize(self.loss, var_list=t_vars)
        """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]      # Discriminator weights
        g_vars = [var for var in t_vars if 'g_' in var.name]      # Generator/Decoder weights
        a_vars = [var for var in t_vars if not 'd_' in var.name]  # Encoder/Decoder weights (all except discriminator)
        e_vars = [var for var in t_vars if 'en_' in var.name]     # Encoder weights
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.d_weight*self.l_rate, epsilon=1e-06) \
                                 .minimize(self.d_loss, var_list=d_vars)  # d_opt  ~ train discriminator
            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.g_weight*self.l_rate, epsilon=1e-06) \
                                 .minimize(self.g_loss, var_list=g_vars)  # g_opt  ~ train generator (adversarial)
            self.g2_opt = tf.train.AdamOptimizer(learning_rate=self.g2_weight*self.l_rate, epsilon=1e-06) \
                                  .minimize(self.loss, var_list=g_vars)   # g2_opt ~ train generator (l2/KL)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.e_weight*self.l_rate, epsilon=1e-06) \
                               .minimize(self.loss, var_list=e_vars)      # e_opt  ~ train encoder (l2/KL)
        

        """ Summary """
        # TensorBoard summaries
        l2_sum = tf.summary.scalar("l2", self.L2_loss)
        reg_sum = tf.summary.scalar("reg", self.Reg_loss)
        reg_sum_scaled = tf.summary.scalar("regscale", self.reg_weight*self.Reg_loss)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        kl_sum_scaled = tf.summary.scalar("klscale", self.kl_weight*self.KL_divergence)
        d_sum = tf.summary.scalar("dloss", self.d_loss)
        loss_sum = tf.summary.scalar("loss", self.loss)
        
        # Merged summary
        self.merged_summary_op = tf.summary.merge_all()

    # Train VAE Model
    def train(self):
        # Initialize variables
        tf.global_variables_initializer().run()

        # Define saver
        self.saver = tf.train.Saver()

        # Define training/validation summary writers
        self.writer = tf.summary.FileWriter(self.log_dir + '/training_' + self.model_name, self.sess.graph)
        self.vwriter = tf.summary.FileWriter(self.log_dir + '/validation_' + self.model_name, self.sess.graph)

        # Restore latest checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / (self.num_batches * self.transformation_count))
            start_batch_id = checkpoint_counter - start_epoch * (self.num_batches * self.transformation_count)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            l_rate = self.get_learning_rate(self.learning_rate)
            print('     (learning_rate = %.6f)\n' %(l_rate))            
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            l_rate = self.learning_rate

            

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # Shuffle data each epoch
            inds = range(0,self.data_X.shape[0])
            np.random.shuffle(inds)
            if self.use_onehot:
                self.data_X = self.data_X[inds,:,:]
            else:
                self.data_X = self.data_X[inds,:]

            # Decay learning rate each epoch                        
            l_rate = self.get_learning_rate(l_rate)
                    
            # Iterate through batches of data
            for idx in range(start_batch_id, self.num_batches):
                # Iterate through transformations of data
                for transformation in self.transformations:
                    
                    # Get batch
                    batch_X, batch_y = self.get_batch(idx, transformation)

                    if self.c_dim == 1:
                        batch_y = batch_y[:,:,:,0]
                        batch_y = np.expand_dims(batch_y,3)
                    # Update weights
                    """
                    _, sum_str, loss, l2_loss, reg_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op,
                                                                                      self.loss, self.L2_loss,
                                                                                      self.Reg_loss, self.KL_divergence],
                                                                                     feed_dict={self.inputs: batch_X,
                                                                                                self.outputs: batch_y,
                                                                                                self.kl_c: self.kl_weight,
                                                                                                self.reg_c: self.reg_weight,
                                                                                                self.is_training: True})
                    """
                    
                    _,__,___,____,sum_str,loss,l2_loss,reg_loss,kl_loss,g_l,d_l = self.sess.run([self.g_opt, self.d_opt,
                                                                                                 self.g2_opt, self.opt,
                                                                                                 self.merged_summary_op,
                                                                                                 self.loss, self.L2_loss,
                                                                                                 self.Reg_loss, self.KL_divergence,
                                                                                                 self.g_loss, self.d_loss],
                                                                                                feed_dict={self.inputs: batch_X,
                                                                                                           self.outputs: batch_y,
                                                                                                           self.kl_c: self.kl_weight,
                                                                                                           self.reg_c: self.reg_weight,
                                                                                                           self.adv_c: self.adv_weight,
                                                                                                           self.l_rate: l_rate,
                                                                                                           self.is_training: True})
                    
                    self.writer.add_summary(sum_str, counter)
                    
                    # Display training status
                    counter += 1
                    #print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, l2: %.4f, reg: %.4f, kl: %.4f  [reg: %.2e / kl: %.2e]" \
                    #      % (epoch, np.mod(counter, self.num_batches*self.transformation_count), \
                    #         self.num_batches * self.transformation_count, time.time() - start_time, \
                    #         loss, l2_loss, self.reg_weight*reg_loss, self.kl_weight*kl_loss, reg_loss, kl_loss))
                    """
                    if np.mod(counter, self.display_step) == 0:
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, l2: %.4f, reg: %.4f, kl: %.4f" \
                              % (epoch, np.mod(counter, self.num_batches*self.transformation_count), \
                                 self.num_batches * self.transformation_count, time.time() - start_time, \
                                 loss, l2_loss, self.reg_weight*reg_loss, self.kl_weight*kl_loss))
                    """
                    if np.mod(counter, self.display_step) == 0:
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, l2: %.4f, g_loss: %.4f, d_loss: %.4f, kl: %.4f" \
                              % (epoch, np.mod(counter, self.num_batches*self.transformation_count), \
                                 self.num_batches * self.transformation_count, time.time() - start_time, \
                                 loss, l2_loss, g_l, d_l, self.kl_weight*kl_loss))
                        

                    # Save predictions
                    if np.mod(counter, self.plot_step) == 0:
                        #pred, soln = self.sess.run([self.masked_pred, self.masked_y],
                        inputs, pred, soln = self.sess.run([self.inputs, self.pred, self.outputs],
                                                           feed_dict={self.inputs: batch_X, self.outputs: batch_y, self.is_training: False})

                        # Save plots
                        for k in range(0,self.plots):
                            prediction = pred[k,:,:,:]
                            soln = soln[k,:,:,:]
                            inputs = inputs[k,:]

                            """
                            # Find Interior Indices
                            ext_indices = (soln == 0.0)
                        
                            #domain_mask = tf.expand_dims(interior_indices,3)
                            z_tensor = 0.0*soln
                            
                            # Masked Versions
                            prediction[ext_indices] = 0.0
                        
                            #print(prediction.shape)
                            pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
                            #print(pred_layered.shape)
                            pred_layered = np.transpose(pred_layered,(1,2,0))
                            
                            # Save Prediction Array
                            #save_dir = check_folder(self.result_dir + '/' + self.model_dir)
                            save_dir = check_folder(self.result_dir)
                            prediction_array_filename = save_dir + '/' + str(counter) + '_prediction.npy'
                            np.save(prediction_array_filename, pred_layered)
                            
                            soln_layered = np.array([soln[:,:,0],soln[:,:,0],soln[:,:,0]])
                            soln_layered = np.transpose(soln_layered,(1,2,0))
                            
                            # Save Solution Array
                            #save_dir = check_folder(self.result_dir + '/' + self.model_dir)
                            save_dir = self.result_dir
                            soln_array_filename = save_dir + '/' + str(counter) + '_solution.npy'
                            np.save(soln_array_filename, soln_layered)
                            """

                            # Save Prediction Array
                            if self.c_dim == 1:
                                pred_layered = np.array([prediction[:,:,0],prediction[:,:,0],prediction[:,:,0]])
                                pred_layered = np.transpose(pred_layered,(1,2,0))
                            else:
                                pred_layered = prediction
                            save_dir = check_folder(self.result_dir)
                            prediction_array_filename = save_dir + '/' + str(counter) + '_prediction.npy'
                            np.save(prediction_array_filename, pred_layered)

                            
                            # Save Solution Array
                            if self.c_dim == 1:
                                soln_layered = np.array([soln[:,:,0],soln[:,:,0],soln[:,:,0]])
                                soln_layered = np.transpose(soln_layered,(1,2,0))
                            else:
                                soln_layered = soln
                            save_dir = self.result_dir
                            soln_array_filename = save_dir + '/' + str(counter) + '_solution.npy'
                            np.save(soln_array_filename, soln_layered)


                    # Validation step
                    if np.mod(counter, self.validation_step) == 0:
                        vbatch_X, vbatch_y = self.get_batch(idx, transformation, validation=True)
                        """
                        pred, soln, loss, l2_loss, reg_loss, kl_loss = self.sess.run([self.pred, self.outputs,
                                                                                      self.loss, self.L2_loss,
                                                                                      self.Reg_loss, self.KL_divergence],
                                                                                     feed_dict={self.inputs: batch_X,
                                                                                                self.outputs: batch_y,
                                                                                                self.kl_c: self.kl_weight,
                                                                                                self.reg_c: self.reg_weight,
                                                                                                self.is_training: False})
                        """
                        pred, soln, loss, l2_loss, reg_loss, kl_loss, adv_loss = self.sess.run([self.pred, self.outputs,
                                                                                                self.loss, self.L2_loss,
                                                                                                self.Reg_loss, self.KL_divergence,
                                                                                                self.Adv_loss],
                                                                                               feed_dict={self.inputs: batch_X,
                                                                                                          self.outputs: batch_y,
                                                                                                          self.kl_c: self.kl_weight,
                                                                                                          self.reg_c: self.reg_weight,
                                                                                                          self.adv_c: self.adv_weight,
                                                                                                          self.l_rate: l_rate,
                                                                                                          self.is_training: False})


                    
                        # Update loss function weights
                        #self.kl_weight = np.min([self.kl_weight, 0.5*l2_loss/kl_loss])
                        #self.kl_weight = np.min([self.kl_weight, 2.0*l2_loss/kl_loss])
                        self.kl_weight = np.min([self.kl_weight, self.kl_loss_tolerance*l2_loss/kl_loss])
                        self.reg_weight = np.min([self.reg_weight, self.loss_tolerance*l2_loss/reg_loss])
                        #self.adv_weight = np.min([self.adv_weight, self.adv_loss_tolerance*l2_loss/adv_loss])
                        
                        print('\n----------------------------------------------------------------------------------')
                        print('|  Validation Loss = %.5f    reg_weight = %.2E     kl_weight = %.4f     |' \
                              %(loss,self.reg_weight,self.kl_weight))
                        print('----------------------------------------------------------------------------------\n')

                        
            # Reset starting batch_id for next epoch
            start_batch_id = 0

            # Save model after each epoch
            self.save(self.checkpoint_dir, counter)

        # Save final model
        self.save(self.checkpoint_dir, counter)

    # Assemble mini-batch with random transformations
    def get_batch(self, idx, transformation=[0,0], validation=False):
        #rotate = transformation[0]  # rotate = np.random.choice([0,1,2,3])
        #flip = transformation[1]    # flip = np.random.choice([0,1])
        if self.use_transformations:
            rotate = np.random.choice([0,1,2,3])
            flip = np.random.choice([0,1])
        else:
            rotate = 0
            flip = 0

        if validation:
            idx = np.mod(idx, self.num_vbatches)
            batch_X = self.vdata_X[idx*self.vbatch_size:(idx+1)*self.vbatch_size]
            batch_y = self.vdata_y[idx*self.vbatch_size:(idx+1)*self.vbatch_size]
        else:
            batch_X = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_y = self.data_y[idx*self.batch_size:(idx+1)*self.batch_size]
            
        #batch_X = np.array([batch_X])
        #batch_X = np.transpose(batch_X, [1,2,3,0])
        
        #batch_y = np.array([batch_y])
        #batch_y = np.transpose(batch_y, [1,2,3,0])

        if self.use_onehot:
            if self.use_transformations:
                # Apply transformations
                for n in range(0,batch_X.shape[0]):
                    vals_X = batch_X[n,:,:]
                    vals_X = np.rot90(vals_X, k=rotate)
                    if flip == 1:
                        vals_X = np.flipud(vals_X)
                    batch_X[n,:,:] = vals_X

                    vals_y = batch_y[n,:,:,0]
                    vals_y = np.rot90(vals_y, k=rotate)
                    if flip == 1:
                        vals_y = np.flipud(vals_y)
                    batch_y[n,:,:,0] = vals_y

                    # Check onehot
                    #print(sum(sum(batch_X[n,:,:])))
            
        #if self.use_onehot:
        #    batch_X = self.make_onehot(batch_X)
        return batch_X, batch_y

    # Convert to onehot 
    def make_onehot(self,batch_X):
        res = self.res
        SCALING = 10e3
        N = batch_X.shape[0]
        tolerance = 0.0089
        [x_min, x_max] = [-0.0074 - tolerance, 0.0074 + tolerance]
        [z_min, z_max] = [-0.0074 - tolerance, 0.0074 + tolerance]
        x_count = res
        z_count = res
        x_grid = SCALING*np.linspace(x_min,x_max,x_count + 1)
        z_grid = SCALING*np.linspace(z_min,z_max,z_count + 1)

        onehot = np.zeros([N,res,res])
        step_size = x_grid[1] - x_grid[0]
        for n in range(0,N):
            for i in range(0,res):
                for j in range(0,res):
                    x_val = batch_X[n,0]
                    z_val = batch_X[n,1]
                    if (x_val >= x_grid[i]) and (x_val < x_grid[i+1]):
                        if (z_val >= z_grid[j]) and (z_val < z_grid[j+1]):
                            onehot[n,i,j] = 1.0

        return onehot

    # Define adaptive weight for regularization
    def get_learning_rate(self, l_rate):
        l_rate = self.learning_decay * l_rate
        """
        start_step = 0
        start_descent = self.learn_startstep
        mid_step = self.learn_midstep
        end_step = self.learn_endstep
        start_val = self.start_rate
        #mid_val = self.start_rate/4.0
        mid_val = self.start_rate/6.0
        #end_val = self.start_rate/8.0
        end_val = self.start_rate/20.0
        if step <= start_descent:
            l_rate = start_val
        elif step <= mid_step:
            l_rate = start_val + (step-start_descent)*(mid_val-start_val)/(mid_step-start_descent)
        elif step <= end_step:
            l_rate = mid_val + (step-mid_step)*(end_val-mid_val)/(end_step-mid_step)
        else:
            l_rate = end_val
        """
        return l_rate

    # Format model directory name
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    # Save checkpoint
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    # Load saved checkpoint if available
    def load(self, checkpoint_dir):
        import re
        print("\n [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [!] No checkpoints found\n")
            return False, 0
