from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import layers
import numpy as np
from random import shuffle
import os
import time
import itertools

# Load 'reader.py' for converting training data
from reader import *

# Load 'convolution_layers.py' where layers are defined
from convolution_layers import *

# Load 'parameters.py' to specify model parameters
from parameters import *

# Define placeholders to store input batches and corresponding predictions
with tf.name_scope('Training_Data'):
    x = tf.placeholder(tf.float32, [None, n_channels_in], name='x')
    y = tf.placeholder(tf.float32, [None, None, None, n_channels_out], name='y')
    template = tf.placeholder(tf.float32, [None, None, n_channels_out], name='template')

# Define placeholders to store regularization parameters
with tf.name_scope('Regularization'):
    kl_wt = tf.placeholder(tf.float32, name='kl_wt')
    reg_wt = tf.placeholder(tf.float32, name='reg_wt')
    drop = tf.placeholder(tf.float32, name='drop')
    training = tf.placeholder(tf.bool, name='training')
    learning_rt = tf.placeholder(tf.float32, name='learning_rt')



#########################################################################################
###             DEFINE THE LAYERS AND PARAMETERS FOR THE NEURAL NETWORK               ###
#########################################################################################
from Layer import Layer            

latent = 50
latent_chans = 16
latent_res = 4

CONV_LAYERS = [Layer('dense','Dense_1', 20, regularize=False, drop_rate=0.0),
               Layer('dense','Dense_2', 30),
               Layer('dense','Dense_3', latent_res*latent_res*latent_chans)]

T_CONV_LAYERS = [Layer('t_incept','t_Incept_1', 16),
                 Layer('t_incept','t_Incept_2', 16),
                 Layer('t_conv','t_Conv_2', 10),
                 Layer('t_conv','t_Conv_3', 1, kernel_size=3,
                       activation=None, regularize=False, drop_rate=0.0)]


#########################################################################################
#########################################################################################


# Sample from multivariate Gaussian
def sampleGaussian(mean, log_sigma):
    scale = 1.0
    epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
    return tf.cond(training, lambda: mean + 1.0 * epsilon * scale*tf.exp(log_sigma), lambda: mean)

# Define encoder
def Encoder(X, reuse=None):
    print(y)
    with tf.name_scope('Encoder'):    

        layer_count = np.shape(CONV_LAYERS)[0]

        for n in range(0,layer_count):
            ## Specify input tensor
            if n == 0:
                Y = X
            
            layer = CONV_LAYERS[n]
            with tf.name_scope(layer.name):
                Y = layer.apply(Y)
            
        return Y


# Define probabalistic/sampling layer
def Probabilistic_Layer(X):
    with tf.name_scope('Sampler'):

        if USE_PROB:
            Z_mean = X[:,0:latent]; Z_log_sigma = X[:,latent:]
            Y = sampleGaussian(Z_mean, Z_log_sigma)
        else:
            Y = X
            Z_mean = tf.zeros_like(Y)
            Z_log_sigma = tf.ones_like(Y)

        return Y, Z_mean, Z_log_sigma


# Define decoder
def Decoder(X,reuse=None):
    with tf.name_scope('Decoder'):
        layer_count = np.shape(T_CONV_LAYERS)[0]

        ## Reshape to square tensor
        if USE_PROB:
            name = 't_Dense_1'

            Y = dense_layer(X, latent_res*latent_res*latent_chans,
                            drop_rate=0.0, training=training, regularize=True, name=name, reuse=reuse)
            Y = tf.reshape(Y, [-1, latent_res, latent_res, latent_chans])
        else:
            X = tf.reshape(X, [-1, latent_res, latent_res, latent_chans])

        for n in range(0,layer_count):
            ## Specify input tensor
            if (n == 0) and (not USE_PROB):
                Y = X

            layer = T_CONV_LAYERS[n]
            with tf.name_scope(layer.name):
                Y = layer.apply(Y)
        
        return Y



###
# Define Convolutional Neural Network Model
###
def VAE_network(X):

    # Note: Input 'X' must be in NHWC format (N, Height, Width, Channel)
    # Channel sizes are specified in the 'parameters.py' file

    # Encode input data:  [None, N, N, 1]  -->  [None, C*n*n] 
    Y = Encoder(X)
    
    # Sample from latent space:  [None, C*n^2]  --> [None, L]
    Y, Z_mean, Z_log_sigma = Probabilistic_Layer(Y)
    
    # Decode latent vector:  [None, L]  -->  [None, N, N, 1]
    Y = Decoder(Y)
    
    return Y, Z_mean, Z_log_sigma



# Define prediction as output of convolutional neural network
#pred, z_mean, z_log_sigma = conv_net(x)
with tf.name_scope('VAE_Net'):
    pred, z_mean, z_log_sigma = VAE_network(x)
    prediction = tf.identity(pred, name='prediction')
    #e = Encoder(x, True)
    #encoded = tf.identity(e, name='encoded')

#with tf.name_scope('Test_Net'):
#    decoded = Decoder(l, True)


###
# Define cost function 
###

print(y)
with tf.name_scope('Cost_Function'):
    
    # Template storing locations of atoms before impact
    template_tiled = tf.tile(tf.expand_dims(template, 0), [train_batch_size,1,1,1])
    
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
    
    # Mean Square Cost Function
    with tf.name_scope('MS_Cost'):
        ms_cost = tf.reduce_sum(tf.reduce_sum(tf.pow(masked_pred-masked_y, 2), axis=[1,2]))
        #ms_cost = tf.reduce_mean(tf.reduce_mean(tf.pow(masked_pred-masked_y, 2), axis=[1,2]))

        ## L1-COST
        #ms_cost = tf.reduce_mean(tf.reduce_mean(tf.abs(pred-y), axis=[1]))
        deriv1_cost = ms_cost
    
    # KL-Divergence Cost Function
    with tf.name_scope('KL_Cost'):

        # Kullback-Leibler divergence
        def kullbackLeibler(mean, log_sigma):
            # = -0.5 * (1 + log(sigma**2) - mean**2 - sigma**2)
            divergence = -0.5 * tf.reduce_mean(1 + 2 * log_sigma - mean**2 - tf.exp(2 * log_sigma), axis=1)
            return divergence

        if USE_PROB:
            kl_cost = kl_wt * tf.reduce_mean(kullbackLeibler(z_mean, z_log_sigma), axis=0)
        else:
            kl_cost = ms_cost

        
    # Regularization Cost Function
    with tf.name_scope('Reg_Cost'):
        reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_count = reg_list
        reg_cost = reg_wt * sum(reg_list)


    with tf.name_scope('Scale_Cost'):
        scale_cost =  tf.reduce_mean( tf.abs( tf.add(tf.reduce_max(tf.abs(pred),axis=1) , -1.0) ) )
        
    # Total Cost Function
    with tf.name_scope('Total_Cost'):
        if USE_PROB:
            cost = cost_weights['ms']*ms_cost + reg_cost + kl_cost
        else:
            cost = cost_weights['ms']*ms_cost + reg_cost
        

# Define update operations for batch normalization    
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Run Adam Optimizer to minimize cost
with tf.control_dependencies(update_ops):
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rt,epsilon=1e-06).minimize(cost)


# Define summary for log file
tf.summary.scalar("MS_Loss", ms_cost)
tf.summary.scalar("Scale_Loss", scale_cost)
tf.summary.scalar("Regularization_Loss", reg_cost)
tf.summary.scalar("Kullback-Leibler_Loss", kl_cost)
tf.summary.scalar("Total_Loss", cost)

# Merge summary data
merged = tf.summary.merge_all()


# Define Initializer
init = tf.global_variables_initializer()

# Train VAE model with validation checks
def train_model(t_indices, v_indices, val_row):
    # Begin TensorFlow Session
    with tf.Session() as sess:
        
        if LOAD_PREVIOUS:
            # Restore Previous Session
            new_saver = tf.train.Saver()
            new_saver.restore(sess, tf.train.latest_checkpoint(prev_checkpoint_directory + str(val_row)))
        else:
            # Initialize Variables
            sess.run(init)
        
        # Define writer for saving training results to log file
        writer = tf.summary.FileWriter(logdir + str(val_row) + '/training', graph=tf.get_default_graph(), max_queue=10)
        
        # Define writer for saving validation results to log file
        vwriter = tf.summary.FileWriter(logdir + str(val_row) + '/validation', graph=tf.get_default_graph(), max_queue=10)
        
        # Define saver for current session
        saver = tf.train.Saver(max_to_keep=1)
        
        # Randomize indices for training batches
        batch_indices = [i for i in range(0,train_batches-1)]
        shuffle(batch_indices)
        
        # Initialize starting time for progress bar
        #start_time = time.time()
        start_time = get_time()
        
        # Initialize step counts
        step = 1; vstep = 0

        # Specify template file for atom positions
        template_file = array_directory + 'template.npy'
        template_array = np.load(template_file)
        template_array = np.array([template_array[:,:,0]])
        template_array = np.transpose(template_array,[1,2,0])

        # Initialize KL-divergence and regularization weights
        kl_c = cost_weights['kl']
        reg_c = cost_weights['reg']

        # Iterate through specified number of epochs
        for n in range(0,epochs):

            # Randomize batches for each epoch
            t_indices = [i for i in range(0,train_count)]
            shuffle(t_indices)
            
            # Iterate through batches and transforamtions
            for K in batch_indices:
                for transform in transformations:

                    # Specify training state and dropout rate
                    train = True; drop_rate = DROP                

                    # Assemble Batch
                    batch_x, batch_y = train_next_batch(K, train_batch_size, t_indices, transformation=transform)
                    #print(batch_x.shape)
                    #print(batch_y.shape)

                    l_rate = get_learning_rate(step)

                    # Run Optimizer and Update Weights
                    if (step % metadata_step == 0) and (step >= LOG_START):

                        # Record meta data (e.g. compute time)
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        
                        # Write training summary to log file
                        summary, _ = sess.run([merged, optimizer],
                                              feed_dict={x: batch_x, y: batch_y, template: template_array,
                                                         reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                         training: train, learning_rt: l_rate},
                                              options=run_options, run_metadata=run_metadata)
                        
                        writer.add_run_metadata(run_metadata, 'step_' + str(step))
                        writer.add_summary(summary, step)
                        writer.flush()

                        # Record meta data (e.g. compute time)
                        vrun_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        vrun_metadata = tf.RunMetadata()

                        # Assemble validation batch and write summary to log file
                        vbatch_x, vbatch_y = train_next_batch(vstep, validation_batch_size, v_indices)
                        train = False; drop_rate = 0.0
                        vsummary = sess.run(merged,
                                            feed_dict={x: vbatch_x, y: vbatch_y, template: template_array,
                                                       reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                       training: train},
                                            options=vrun_options, run_metadata=vrun_metadata)
                        
                        vwriter.add_run_metadata(vrun_metadata, 'step_' + str(step))
                        vwriter.add_summary(vsummary, step)
                        vwriter.flush()

                        # Increment validation batch step
                        vstep = (vstep + 1)%validation_batches

                    elif (step % log_step == 0) and (step >= LOG_START):

                        # Write training summary to log file
                        summary, _ = sess.run([merged, optimizer],
                                              feed_dict={x: batch_x, y: batch_y, template: template_array,
                                                         reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                         training: train, learning_rt: l_rate})
                        
                        writer.add_summary(summary, step)
                        writer.flush()

                        # Assemble validation batch and write summary to log file
                        vbatch_x, vbatch_y = train_next_batch(vstep, validation_batch_size, v_indices)
                        train = False; drop_rate = 0.0
                        vsummary = sess.run(merged,
                                            feed_dict={x: vbatch_x, y: vbatch_y, template: template_array,
                                                       reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                       training: train})
                        
                        vwriter.add_summary(vsummary, step)
                        vwriter.flush()

                        # Increment validation batch step
                        vstep = (vstep + 1)%validation_batches

                    else:
                        sess.run(optimizer,
                                 feed_dict={x: batch_x, y: batch_y, template: template_array,
                                            reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                            training: train, learning_rt: l_rate})




                    # Display Minibatch Loss
                    if step % display_step == 0:
                        ms_loss, kl_loss, der_loss, reg_loss = sess.run([ms_cost, kl_cost, deriv1_cost, reg_cost],
                                                                        feed_dict={x: batch_x, y: batch_y, template: template_array,
                                                                                   reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                                                   training: train})
                        # Display progress bar defined in 'parameters.py' file
                        d_str, _ = display_progress(start_time,total_batches,step,n,ms_loss,kl_loss,der_loss,reg_loss,val_row)
                        print(d_str)

                    # Display min/max values and plot prediction
                    if step % plot_step == 0:
                        [vals, in_data, out_data] = sess.run([masked_pred, x, masked_y],
                                                             feed_dict={x: batch_x, y: batch_y, template: template_array,
                                                                        reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                                        training: train})
                        prediction = vals[0,:,:,:]
                        soln = out_data[0,:,:,:]
                        
                        # Get current indices
                        current_ID = t_indices[K*train_batch_size]
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
                        save_dir = prediction_subdir + str(val_row) + '/'
                        prediction_array_filename = save_dir + str(step) + '_prediction_' + ID_label + '.npy'
                        np.save(prediction_array_filename, pred_layered)
                        
                        soln_layered = np.array([soln[:,:,0],soln[:,:,0],soln[:,:,0]])
                        soln_layered = np.transpose(soln_layered,(1,2,0))
                        
                        # Save Solution Array
                        soln_array_filename = save_dir + str(step) + '_solution_' + ID_label + '.npy'
                        np.save(soln_array_filename, soln_layered)
                        
                        

                    # Display Validation Loss
                    if step % validation_step == 0:

                        # Assemble Validation Batch
                        vbatch_x, vbatch_y = train_next_batch(vstep, validation_batch_size, v_indices)
                        train = False; drop_rate = 0.0
                        vms_loss, vreg_loss, vkl_loss = sess.run([ms_cost, reg_cost, kl_cost],
                                                      feed_dict={x: vbatch_x, y: vbatch_y, template: template_array,
                                                                 reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                                 training: train})

                        #kl_c = np.min([kl_c, 0.5*vms_loss*kl_c/vkl_loss])
                        #reg_c = np.min([reg_c, 0.5*vms_loss*reg_c/vreg_loss])
                        kl_c = np.min([kl_c, 0.5*vms_loss*kl_c/vkl_loss])
                        reg_c = np.min([reg_c, 0.75*vms_loss*reg_c/vreg_loss])

                        if USE_PROB:
                            print('\n------------------------------------------------------------------------')
                            print('|  Validation Loss = %.5f    reg_c = %.2E     kl_c = %.4f     |' %(vms_loss,reg_c,kl_c))
                            print('------------------------------------------------------------------------\n')
                        else:
                            print('\n-------------------------------------------------------')
                            print('|    Validation Loss = %.5f                        |' %(vms_loss))
                            print('-------------------------------------------------------\n')

                    

                    # Check for Early Stopping
                    if (step % stopping_step == 0) and (step > STOPPING_START):

                        # Assemble Validation Batch
                        vbatch_x, vbatch_y = train_next_batch(0, stopping_batch_size, v_indices)
                        train = False
                        drop_rate = 0.0
                        vms_loss = sess.run(ms_cost,
                                            feed_dict={x: vbatch_x, y: vbatch_y, template: template_array,
                                                       reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                                       training: train})
                        
                        print('\n------------------------------------------')
                        print('|    Stopping Loss = %.5f              |' %(vms_loss))
                        print('------------------------------------------\n\n')

                        with open(model_dir + 'stopping_losses.csv', 'a') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter='\t')
                            csvwriter.writerow([str(val_row), step, vms_loss])

                        
                        if vms_loss < stopping_level:
                            save_file = save_directory + str(val_row) + '/' + model_name
                            saver.save(sess, save_file, global_step=step)
                            print('\n\nTraining ' + str(val_row) +   ' complete. Model has been saved to:')
                            print(save_directory + str(val_row))
                            print('')
                            return start_time, step
                    
                    # Save Model Checkpoint
                    if step % save_step == 0:
                        save_file = save_directory + str(val_row) + '/' + model_name
                        saver.save(sess, save_file, global_step=step)

                        
                    # Increment training batch step
                    step += 1


        # Save final checkpoint
        save_file = save_directory + str(val_row) + '/' + model_name
        saver.save(sess, save_file, global_step=step)
        print('\n\nTraining ' + str(val_row) +   ' complete. Model has been saved to:')
        print(save_directory + str(val_row))
        print('')
        
    return start_time, step
            

# Computes cross-validation accuracy of trained model
def validate_model(v_indices,val_row):
    # Begin TensorFlow Session
    with tf.Session() as sess:

        print('\nComputing Cross Validation ' + str(val_row) +   ' accuracy...')

        # Restore Previous Session
        new_saver = tf.train.Saver()
        new_saver.restore(sess, tf.train.latest_checkpoint(prev_checkpoint_directory + str(val_row)))
        
        # Compute final validation accuracy
        validation_loss = 0.0
        for n in range(0,validation_count):
            
            # Assemble Validation Batch
            vbatch_x, vbatch_y = train_next_batch(n, 1, v_indices)
            train = False; drop_rate = 0.0; reg_c = 0.0; kl_c = 0.0
            vms_loss = sess.run(ms_cost,
                                feed_dict={x: vbatch_x, y: vbatch_y, template: template_array,
                                           reg_wt: reg_c, kl_wt: kl_c, drop: drop_rate,
                                           training: train})
            
            validation_loss += vms_loss

        average_loss = validation_loss/validation_count
        
        with open(model_dir + 'cv_accuracy.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t')
            csvwriter.writerow([str(val_row) , average_loss])
        
        print('\nCV Accuracy = %.6f' %(average_loss))
        print('\n\n')

        
# Draw random samples from the latent variable space
#def sample_latent(samples=10, val_row=1):
#    with tf.Session() as sess:
#
#        # Restore Previous Session
#        new_saver = tf.train.Saver()
#        new_saver.restore(sess, tf.train.latest_checkpoint(prev_checkpoint_directory + str(val_row)))
#
#        sample_directory = model_dir + 'Samples/'
#        if not os.path.exists(sample_directory):
#            os.makedirs(sample_directory)
#
#        file_prefix = sample_directory + 'sample_'
#
#        rand_vals = np.random.randn(samples,latent)
#        
#        for n in range(0,samples):
#            filename = file_prefix + str(n) + '.npy'
#            l_vals = rand_vals[n,:]
#            batch_l = [l_vals]
#            vals = sess.run(decoded, feed_dict={l: batch_l,
#                                                reg_wt: 0.0,kl_wt: 0.0, drop: 0.0,
#                                                training: False})
#            np.save(filename, vals)






# Count total number of model variables/parameters
def count_variables():
    with tf.Session() as sess:

        # Restore Previous Session
        new_saver = tf.train.Saver()

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        
        return total_parameters
                            



# Count total number of model variables/parameters
def list_variables():
    with tf.Session() as sess:

        # Restore Previous Session
        new_saver = tf.train.Saver()

        for variable in tf.get_default_graph().as_graph_def().node:
            print(variable.name)
