from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from random import shuffle

import os
import time

# Load 'reader.py' for converting images
from encoder import *
from parameters import *
from convolution_layers import *

# Determine batch sizes
#data_batches = data_count//batch_size
data_batches = train_count//batch_size

# Determine total number of batches
transformations_array = np.array(transformations)
transformations_count = (transformations_array.shape)[0]
total_batches = data_batches*transformations_count*epochs


learn_decay_rate = 0.95


# Define placeholders to store input batches and corresponding predictions
with tf.name_scope('Training_Data'):
    x = tf.placeholder(tf.float32, [None, n_channels_in], name='x')
    y = tf.placeholder(tf.float32, [None, None, None, n_channels_out], name='y')
    template = tf.placeholder(tf.float32, [None, None, n_channels_out], name='template')
    learning_rt = tf.placeholder(tf.float32, name='learning_rt')
    

# Specify Intermediate Channel Sizes and Node Counts
#nodes = [16, 32, 16*16*resolution]
middle_channels = 64
middle_res = 8
nodes = [10, 10, 25, 50, middle_res*middle_res*middle_channels]
channels = [64, 32, 16, n_channels_out]

# Specify Kernel/Filter Sizes
kernels = [3, 3, 3]

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
    c_ind = 0; channel_count = channels[c_ind]
    k_ind = 0; kernel_size = kernels[k_ind]
    #Y = transpose_inception_v3(Y, channel_count, stride=1, training=training)
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=1, training=training)
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=2, training=training)

    # [16, 16]  -->  [32, 32]
    c_ind += 1; channel_count = channels[c_ind]
    k_ind += 1; kernel_size = kernels[k_ind]
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=2, training=training)

    # [32, 32]  -->  [64, 64]
    channel_count = n_channels_out
    k_ind += 1; kernel_size = kernels[k_ind]
    Y = transpose_conv2d_layer(Y, channel_count, kernel_size, stride=2, activation=None, add_bias=True, regularize=False, drop_rate=0.0, batch_norm=False,training=training)

    Y = tf.nn.sigmoid(Y)

    return Y


# Define prediction from convolutional neural network
#pred = conv_net(x)
with tf.name_scope('VAE_Net'):
    pred = conv_net(x)
    pred = tf.identity(pred, name='prediction')


# Determine TensorFlow Version
tf_version = tf.__version__

# Template storing locations of atoms before impact
template_tiled = tf.tile(tf.expand_dims(template, 0), [batch_size,1,1,1])

# Find Interior Indices
interior_indices = tf.not_equal(template_tiled,0.0)

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


## Masked prediction node
with tf.name_scope('VAE_Net_Masked'):
    masked_pred = tf.identity(masked_pred, name='masked_prediction')
    masked_y = tf.identity(masked_y, name='masked_y')

# Mean Square Cost Function
with tf.name_scope('MS_Cost'):
    ms_cost = tf.reduce_sum(tf.reduce_sum(tf.pow(masked_pred-masked_y, 2), axis=[1,2]))
    #ms_cost = tf.reduce_mean(tf.reduce_mean(tf.pow(masked_pred-masked_y, 2), axis=[1,2]))


# Total Cost Function
with tf.name_scope('Total_Cost'):
    cost = ms_cost

# Run Adam Optimizer to minimize cost
with tf.name_scope('Optimizer'):
    if TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-06).minimize(cost)


# Define summary of cost for log file
tf.summary.scalar("MS_Loss", ms_cost)
tf.summary.scalar("Total_Loss", cost)


# Merge summary data
merged = tf.summary.merge_all()

# Define Initializer
init = tf.global_variables_initializer()


# Begin TensorFlow Session
with tf.Session() as sess:
    
    if LOAD_PREVIOUS:
        # Restore Previous Session
        new_saver = tf.train.Saver()
        new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_directory))
    else:
        # Initialize Variables
        sess.run(init)
        
    # Define writer for log file
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph(), max_queue=10) 

    # Define saver for current session
    saver = tf.train.Saver(max_to_keep=3)
    
    
    
    # Step through each batch
    # learning from transformed versions sequentially.
    step = 1
    start_time = time.clock()
    progress = '   [  Estimated Time  ~  N/A  ]'

    template_file = array_directory + 'template.npy'
    template_array = np.load(template_file)
    template_array = np.array([template_array[:,:,0]])
    template_array = np.transpose(template_array,[1,2,0])


    # RUN TO TEST
    #sess.run(pred, feed_dict={x: batch_x, y: batch_y, template: template_array})

    # Save final checkpoint
    saver.save(sess, checkpoint_directory + 'FINAL_MASKED', global_step=0)
        
    l_rate = learning_rate
    
    for n in range(0,epochs):
        # Randomize Batches
        #data_indices = [i for i in range(1,data_count+1)]
        #shuffle(data_indices)
        data_indices = train_indices
        shuffle(data_indices)

        if (n+1) % 4 == 0:
            l_rate = learn_decay_rate*l_rate

        
        # Define indices to iterate through
        indices = range(1,data_batches)
        
        # Iterate through batches and transformations
        for M in indices:
            for transform in transformations:
                # Assemble Batch
                batch_x, batch_y = train_next_batch(M, data_indices, data_batch_size, transform)
                
                # Run Optimizer and Update Weights
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, template: template_array, learning_rt: l_rate})
                
                # Display Minibatch Loss
                if step % display_step == 0:
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, template: template_array, learning_rt: l_rate})
                    display_progress(loss, start_time, step, total_batches, n)
                    
                # Plot Prediction
                """
                if step % plot_step == 0:
                    #[vals, in_data, out_data] = sess.run([pred, x, y], feed_dict={x: batch_x, y: batch_y})
                    #[vals, in_data, out_data] = sess.run([masked_pred, x, y], feed_dict={x: batch_x, y: batch_y, template: template_array})
                    [vals, in_data, out_data] = sess.run([masked_pred, x, masked_y], feed_dict={x: batch_x, y: batch_y, template: template_array})
                    prediction = vals[0,:,:,:]
                    soln = out_data[0,:,:,:]

                    # Get current indices
                    current_ID = data_indices[M*data_batch_size]
                    ID_label = str(current_ID)
                    
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
                    prediction_array_filename = prediction_dir + str(step) + '_prediction_' + ID_label + '.npy'
                    np.save(prediction_array_filename, pred_layered)

                    soln_layered = np.array([soln[:,:,0],soln[:,:,0],soln[:,:,0]])
                    soln_layered = np.transpose(soln_layered,(1,2,0))

                    # Save Solution Array
                    soln_array_filename = prediction_dir + str(step) + '_solution_' + ID_label + '.npy'
                    np.save(soln_array_filename, soln_layered)
                """    

                # Save Model
                if step % save_step == 0:
                    summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y, template: template_array})
                    saver.save(sess, save_file, global_step=step)
                    writer.add_summary(summary, step)
                    #saver.save(sess, save_file, global_step=step+current_global_step)
                    #writer.add_summary(summary, step + current_global_step)
                    
                step += 1
                
    # Save final checkpoint
    saver.save(sess, save_file, global_step=step)
    
    # Save current global step
    with open('./current_global_step.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow([str(step)])  
        #csvwriter.writerow([str(step+current_global_step)])    
        
    print('\n\nTraining complete. Results have been saved to:')
    print(checkpoint_directory)
    print('\n\n')
