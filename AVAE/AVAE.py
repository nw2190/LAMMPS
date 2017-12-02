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
        self.start_rate = 0.00015
        self.learn_startstep = int(40*750//45)
        #self.learn_midstep = int(40*1500//45)
        self.learn_midstep = int(40*2500//45)
        #self.learn_endstep = int(100*1500//45)
        self.learn_endstep = int(40*7500//45)
        self.learning_rate = self.start_rate

        # Validation batches
        self.validation_step = 16
        self.vbatch_size = self.batch_size

        # Early stopping criterion
        self.stopping_step = 100000
        self.stopping_size = self.batch_size
        self.stopping_level = 0.001

        # Latent connections for decoder
        self.latent_res = 4
        self.latent_chans = 50

        # Loss function weights
        self.loss_tolerance = 0.5
        #self.adv_loss_tolerance = 0.1
        ##self.adv_loss_tolerance = 0.25
        self.adv_loss_tolerance = 0.175 
        self.l2_weight = 1.0
        self.reg_weight = 0.00005
        self.kl_weight = 0.5
        self.adv_weight = 0.5

        # Specify start of adversarial training
        #self.disc_start = 2500
        #self.adv_start = 5000
        #self.adv_end = 15000
        ##self.disc_start = 1000
        ##self.adv_start = 2000
        ##self.adv_end = 15000
        self.disc_start = 1250
        self.adv_start = 3000
        self.adv_end = 15000
                                        

        template = np.load('./Arrays/template.npy')
        template = np.array([template[:,:,0]])
        self.template = np.transpose(template,[1,2,0])
        
        # Specify number of plots to save
        self.plots = 1
        self.plot_step = 10

        # channel count
        self.c_dim = 1
        
        # latent vector dimension
        self.z_dim = z_dim
        
        # load input data
        data_X = np.load('./DATA.npy')
        total_size = len(data_X)
        input_scaling = 0.1
        self.data_X = input_scaling*data_X[:int(4*total_size//5)]
        self.vdata_X = input_scaling*data_X[int(4*total_size//5):]
        
        # load targets
        data_y = np.load('./TARGETS.npy')
        self.data_y = data_y[:int(4*total_size//5)]
        self.vdata_y = data_y[int(4*total_size//5):]
        
        # specify transformations
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

            net = dense_layer(x, 20, training=self.is_training,regularize=False, name='dense_1')
            
            #net = dense_layer(net, 20, training=self.is_training,regularize=False, name='dense_2')

            # Dense encoding layer
            gaussian_params = dense_layer(net, 2*self.z_dim, training=self.is_training,
                                          regularize=False, activation = None, name='dense_enc')
            
            # Gaussian parameters
            mean = gaussian_params[:, :self.z_dim]
            logsigma = gaussian_params[:, self.z_dim:]
            
        return mean, logsigma

    # Decoder
    def decoder(self, z, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):
            # Dense decoding layer
            net = dense_layer(z, self.latent_res*self.latent_res*self.latent_chans,
                              regularize=False, training=self.is_training, name='dense_dec')
            # Reshape
            net = tf.reshape(net, [-1, self.latent_res, self.latent_res, self.latent_chans])

            # Res/16
            #net = transpose_conv2d_layer(net, 96, kernel_size=2, stride=2, training=self.is_training, name='tconv_1', reuse=reuse)
            net = transpose_inception_v3(net, 96, stride=2, training=self.is_training, name='tincept_0', reuse=reuse)

            # Res/8
            net = transpose_inception_v3(net, 64, stride=2, training=self.is_training, name='tincept_1', reuse=reuse)

            # Res/4
            #net = transpose_conv2d_layer(net, 32, kernel_size=3, stride=2, training=self.is_training, name='tconv_2', reuse=reuse)
            net = transpose_inception_v3(net, 32, stride=2, training=self.is_training, name='tincept_2', reuse=reuse)

            # Res/2
            net = transpose_conv2d_layer(net, 1, kernel_size=3, stride=2, activation=tf.nn.sigmoid,
                                         add_bias=False, regularize=False, training=self.is_training, name='tconv_3', reuse=reuse)

            out = net
            return out


    # Discriminator
    def discriminator(self, x, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):

            # Rescale input
            scaling = tf.reduce_max(tf.abs(x), axis=[1,2,3], keep_dims=True)
            ones = tf.ones_like(x)
            scaling = tf.multiply(scaling,ones)
            x = tf.divide(x, scaling)
            self.scaling = scaling

            # Res
            net = conv2d_layer(x, 16, kernel_size=3, regularize=False, training=self.is_training, name='d_conv_1')
            net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last', name='d_maxp_1')

            # Res/2
            net = conv2d_layer(net, 32, kernel_size=3, regularize=False, training=self.is_training, name='d_conv_2')
            net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last', name='d_maxp_2')

            # Res/4
            #net = inception_v3(net, 64, stride=1, regularize=False, training=self.is_training, name='d_incept_1')
            net = conv2d_layer(net, 64, kernel_size=2, activation=None, training=self.is_training, name='d_conv_3')
            net = layers.max_pooling2d(net, 2, 2, padding='same', data_format='channels_last', name='d_maxp_3')

            # Final resolution: Res/8

            # Reshape
            final_res = self.res//np.power(2,3)
            final_channels = 64
            vect_size = int(final_res*final_res*final_channels)
            net = tf.reshape(net, [-1,vect_size])
            
            # Dense layers
            net = dense_layer(net, 2*self.z_dim, regularize=False, training=self.is_training, name='d_dense_1')
            net = dense_layer(net, 100, regularize=False, training=self.is_training, name='d_dense_2')
            net = dense_layer(net, 10, regularize=False, training=self.is_training, name='d_dense_3')
            
            out_logit = linear(net, 1, scope='d_dense_4')
            out = tf.nn.sigmoid(out_logit)
            
        return out, out_logit

        
    # Build TensorFlow Graph
    def build_model(self):
        # data dimensions
        in_dims = [1]
        data_dims = [self.input_height, self.input_width, self.c_dim]

        """ Graph Input """
        # Input/output placeholders
        self.inputs = tf.placeholder(tf.float32, [None] + in_dims, name='inputs')
        self.outputs = tf.placeholder(tf.float32, [None] + data_dims, name='outputs')

        # Training/reuse placeholders
        self.is_training = tf.placeholder(tf.bool, name='training')
        self.reuse = tf.placeholder(tf.bool, name='reuse')

        # Placeholders for adaptive loss function weights
        self.kl_c = tf.placeholder(tf.float32, name='kl_c')
        self.reg_c = tf.placeholder(tf.float32, name='reg_c')
        self.adv_c = tf.placeholder(tf.float32, name='adv_c')        
        
        """ Loss Function """
        # Encode
        self.mu, self.logsigma = self.encoder(self.inputs)
        self.sigma = tf.exp(self.logsigma)
        
        # Sample in latent space
        z = self.sampleGaussian(self.mu,self.logsigma)

        # Decode
        out = self.decoder(z)
        self.pred = out


        # Template storing locations of atoms before impact
        template_tiled = tf.tile(tf.expand_dims(self.template, 0), [self.batch_size,1,1,1])
        
        # Find Interior Indices
        interior_indices = tf.not_equal(template_tiled,0.0)
        
        # Note: solutions are set to 0.0001 the locations of undamaged atoms
        empty_indices = tf.equal(self.outputs,0.0)
        empty_int_indices = tf.logical_and(interior_indices, empty_indices)
        
        zero_tensor = tf.zeros_like(self.outputs)
        eps_tensor = tf.add(zero_tensor, 0.0001)
        ones_tensor = tf.add(zero_tensor, 1.0)
        
        # Masked Versions
        masked_pred = tf.where(interior_indices, tf.maximum(self.pred,eps_tensor), zero_tensor)
        self.masked_pred = tf.minimum(masked_pred,ones_tensor)
        
        # Remove 'displaced' atoms from solution
        masked_y = tf.where(interior_indices, self.outputs, zero_tensor)
        
        # Declare 'displaced' atoms to be fully damaged
        self.masked_y = tf.where(empty_int_indices, ones_tensor, masked_y)
        
        # Mean Square Cost Function
        #self.L2_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(self.masked_pred-self.masked_y, 2), axis=[1,2]))
        self.L2_loss = tf.reduce_mean(tf.reduce_mean(tf.pow(self.masked_pred-self.masked_y, 2), axis=[1,2]))
            
            



        """ Adversarial Loss Function """
        # output of D for real data
        D_real, D_real_logits = self.discriminator(self.masked_y)

        # output of D for fake data
        D_fake, D_fake_logits = self.discriminator(self.masked_pred, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.Adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))



        
            
        # Regularization loss
        self.reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.Reg_loss = sum(self.reg_list)

        # KL loss
        self.KL_divergence = -0.5 * tf.reduce_mean(1 + 2 * self.logsigma - self.mu**2 - tf.exp(2 * self.logsigma), axis=1)
        self.KL_divergence = tf.reduce_mean(self.KL_divergence)

        # Total loss
        #self.loss = self.l2_weight*self.L2_loss + self.kl_c*self.KL_divergence + self.reg_c*self.Reg_loss

        # Total loss
        self.loss = self.l2_weight*self.L2_loss + self.kl_c*self.KL_divergence + self.reg_c*self.Reg_loss + self.adv_c*self.Adv_loss
        
        """ Training """
        # Optimizer
        #t_vars = tf.trainable_variables()
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-06).minimize(self.loss, var_list=t_vars)

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if not 'd_' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-06) \
                    .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-06) \
                    .minimize(self.loss, var_list=g_vars)
            
        
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
            l_rate = self.get_learning_rate(0)
            print('     (learning_rate = %.6f)\n' %(l_rate))            
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1


        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # Iterate through batches of data
            for idx in range(start_batch_id, self.num_batches):
                # Iterate through transformations of data
                for transformation in self.transformations:
                    self.learning_rate = self.get_learning_rate(counter)

                    # Get batch
                    batch_X, batch_y = self.get_batch(idx, transformation)


                    # Update weights
                    # No adversarial training  """ 0 < counter < disc_start """
                    if counter < self.disc_start:
                        a_l = 0.0
                        d_l = 0.0
                        _, summary_str,loss,l2_loss,reg_loss,kl_loss = self.sess.run([self.g_optim, self.merged_summary_op,
                                                                                      self.loss, self.L2_loss,
                                                                                      self.Reg_loss, self.KL_divergence],
                                                                                     feed_dict={self.inputs: batch_X,
                                                                                                self.outputs: batch_y,
                                                                                                self.kl_c: self.kl_weight,
                                                                                                self.reg_c: self.reg_weight,
                                                                                                self.adv_c: 0.0,
                                                                                                self.is_training: True})
                    # Begin training discriminator  """ disc_start < counter < adv_start """
                    elif counter < self.adv_start:
                        a_l = 0.0
                        _, __,summary_str,loss,l2_loss,reg_loss,kl_loss,d_l = self.sess.run([self.g_optim, self.d_optim,
                                                                                             self.merged_summary_op,
                                                                                             self.loss, self.L2_loss,
                                                                                             self.Reg_loss, self.KL_divergence,
                                                                                             self.d_loss],
                                                                                            feed_dict={self.inputs: batch_X,
                                                                                                       self.outputs: batch_y,
                                                                                                       self.kl_c: self.kl_weight,
                                                                                                       self.reg_c: self.reg_weight,
                                                                                                       self.adv_c: 0.0,
                                                                                                       self.is_training: True})

                    # Start adversarial training  """ adv_start < counter < adv_end """
                    elif counter < self.adv_end:
                        _, __,summary_str,loss,l2_loss,reg_loss,kl_loss,a_l,d_l = self.sess.run([self.g_optim, self.d_optim,
                                                                                                 self.merged_summary_op,
                                                                                                 self.loss, self.L2_loss,
                                                                                                 self.Reg_loss, self.KL_divergence,
                                                                                                 self.Adv_loss, self.d_loss],
                                                                                                feed_dict={self.inputs: batch_X,
                                                                                                           self.outputs: batch_y,
                                                                                                           self.kl_c: self.kl_weight,
                                                                                                           self.reg_c: self.reg_weight,
                                                                                                           self.adv_c: self.adv_weight,
                                                                                                           self.is_training: True})
                    # End adversarial training  """ adv_end < counter """
                    else:
                        a_l = 0.0
                        d_l = 0.0
                        _, summary_str,loss,l2_loss,reg_loss,kl_loss = self.sess.run([self.g_optim, self.merged_summary_op,
                                                                                      self.loss, self.L2_loss,
                                                                                      self.Reg_loss, self.KL_divergence],
                                                                                     feed_dict={self.inputs: batch_X,
                                                                                                self.outputs: batch_y,
                                                                                                self.kl_c: self.kl_weight,
                                                                                                self.reg_c: self.reg_weight,
                                                                                                self.adv_c: 0.0,
                                                                                                self.is_training: True})
                    

                    
                    # Update weights
                    #_, summary_str, loss, l2_loss, reg_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op,
                    #                                                                  self.loss, self.L2_loss,
                    #                                                                  self.Reg_loss, self.KL_divergence],
                    #                                                                 feed_dict={self.inputs: batch_X,
                    #                                                                            self.outputs: batch_y,
                    #                                                                            self.kl_c: self.kl_weight,
                    #                                                                            self.reg_c: self.reg_weight,
                    #                                                                            self.is_training: True})
                    
                    self.writer.add_summary(summary_str, counter)

                    # Display training status
                    counter += 1
                    #print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, l2: %.4f, reg: %.4f, kl: %.4f  [reg: %.2e / kl: %.2e]" \
                    #      % (epoch, np.mod(counter, self.num_batches*self.transformation_count), \
                    #         self.num_batches * self.transformation_count, time.time() - start_time, \
                    #         loss, l2_loss, self.reg_weight*reg_loss, self.kl_weight*kl_loss, reg_loss, kl_loss))
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, l2: %.4f, reg: %.4f, kl: %.4f" \
                          % (epoch, np.mod(counter, self.num_batches*self.transformation_count), \
                             self.num_batches * self.transformation_count, time.time() - start_time, \
                             loss, l2_loss, self.reg_weight*reg_loss, self.kl_weight*kl_loss))


                    # Save predictions
                    if np.mod(counter, self.plot_step) == 0:
                        pred, soln = self.sess.run([self.masked_pred, self.masked_y],
                                                   feed_dict={self.inputs: batch_X, self.outputs: batch_y, self.is_training: False})

                        # Save plots
                        for k in range(0,self.plots):
                            prediction = pred[k,:,:,:]
                            soln = soln[k,:,:,:]
                        
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
                            

                    # Validation step
                    if np.mod(counter, self.validation_step) == 0:
                        vbatch_X, vbatch_y = self.get_batch(idx, transformation, validation=True)
                        pred, soln, loss, l2_loss, reg_loss, kl_loss, adv_loss = self.sess.run([self.pred, self.outputs,
                                                                                                self.loss, self.L2_loss,
                                                                                                self.Reg_loss, self.KL_divergence,
                                                                                                self.Adv_loss],
                                                                                               feed_dict={self.inputs: batch_X,
                                                                                                          self.outputs: batch_y,
                                                                                                          self.kl_c: self.kl_weight,
                                                                                                          self.reg_c: self.reg_weight,
                                                                                                          self.adv_c: self.adv_weight,
                                                                                                          self.is_training: False})
                        
                    
                        # Update loss function weights
                        self.kl_weight = np.min([self.kl_weight, self.loss_tolerance*l2_loss/kl_loss])
                        self.reg_weight = np.min([self.reg_weight, self.loss_tolerance*l2_loss/reg_loss])
                        self.adv_weight = np.min([self.adv_weight, self.adv_loss_tolerance*l2_loss/adv_loss])

                        print('\n----------------------------------------------------------------------------------')
                        print('|  Validation Loss = %.5f    reg_weight = %.2E     kl_weight = %.4f     |' \
                              %(loss,self.reg_weight,self.kl_weight))
                        print('----------------------------------------------------------------------------------\n')


                    # Check for early stopping
                    if np.mod(counter, self.stopping_step) == 0:
                        sbatch_X = np.array(self.vdata_X[:self.stopping_size])
                        print(sbatch_X.shape)
                        #sbatch_X = np.array([self.vdata_X[:self.stopping_size,:,:]])
                        #sbatch_X = np.transpose(sbatch_X, [1,2,3,0])

                        sbatch_y = np.array(self.vdata_y[:self.stopping_size,:,:])
                        print(sbatch_y.shape)
                        #sbatch_y = np.transpose(sbatch_y, [1,2,3,0])
                        loss, l2_loss = self.sess.run([self.loss, self.L2_loss], feed_dict={self.inputs: sbatch_X,
                                                                                            self.outputs: sbatch_y,
                                                                                            self.kl_c: self.kl_weight,
                                                                                            self.reg_c: self.reg_weight,
                                                                                            self.adv_c: self.adv_weight,
                                                                                            self.is_training: False})

                        if l2_loss < self.stopping_level:
                            # Save early stopped model
                            self.save(self.checkpoint_dir, counter)
                            print('\n [*] EARLY STOPPING CRITERION MET  \n')
                            return 0

                        

                        
            # Reset starting batch_id for next epoch
            start_batch_id = 0

            # Save model after each epoch
            self.save(self.checkpoint_dir, counter)

        # Save final model
        self.save(self.checkpoint_dir, counter)

    # Assemble mini-batch with random transformations
    def get_batch(self, idx, transformation=[0,0], validation=False):
        rotate = transformation[0]  # rotate = np.random.choice([0,1,2,3])
        flip = transformation[1]    # flip = np.random.choice([0,1])

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

        # Apply transformations
        """
        for n in range(0,batch_X.shape[0]):
            vals_X = batch_X[n,:,:,0]
            vals_X = np.rot90(vals_X, k=rotate)
            if flip == 1:
                vals_X = np.flipud(vals_X)
            batch_X[n,:,:,0] = vals_X

            vals_y = batch_y[n,:,:,0]
            vals_y = np.rot90(vals_y, k=rotate)
            if flip == 1:
                vals_y = np.flipud(vals_y)
            batch_y[n,:,:,0] = vals_y
        """
        
        return batch_X, batch_y

    # Define adaptive weight for regularization
    def get_learning_rate(self, step):
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
