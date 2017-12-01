from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shutil import copyfile
import string
import time
import csv
import os


# Specify current data available
data_count = 40000

# Specify training resolution
train_resolution = 64

# Specify number of channels expected in/out
n_channels_in, n_channels_out = [1, 1]
output_layers = 3

# Specify current epoch number and number of epochs to run
starting_epoch = 1
epochs = 10

# Specify batch sizes
train_batch_size = 30
validation_batch_size = 30

# Specify initial learning rate
#learning_rate = 0.0000075*train_batch_size
#learning_rate = 0.0000125*train_batch_size
learning_rate = 0.00001*train_batch_size
#learning_rate = 0.000025*train_batch_size

# Specify whether to use probabilistic model
USE_PROB = False

# Define weights for cost function
if USE_PROB:
    cost_weights = {'ms' : 1.0, 'deriv1' : 0.0, 'kl' : 0.01, 'reg' : 0.00005, 'scaling' : 0.025}
else:
    cost_weights = {'ms' : 1.0, 'deriv1' : 0.1, 'kl' : 0.0, 'reg' : 0.00005}
    
# Specify latent parameters and channel count
latent = 100
#latent_channels = 200
#latent = 140
latent_channels = 50

# Specify intermediate channel sizes
channels = [12, 24, 48, 96, 96, 48, 24, 12]
#channels = [6, 12, 24, 48, 48, 24, 12, 6]
#channels = [16, 32, 64, 96, 96, 64, 32, 16]
#channels = [16, 32, 64, 128, 128, 64, 32, 16]

# Specify kernel/filter sizes
kernels = [3, 3, 2, 2, 2, 2, 3, 3]

# Specify dropout rate
DROP = 0.0

# Specify batch normalization usage
BATCH_NORM = False

# Define adaptive weight for regularization
def get_learning_rate(step):
    start_step = 0
    start_descent = 40*750//train_batch_size
    end_step = 40*1500//train_batch_size
    start_val = learning_rate
    end_val = learning_rate/4.0
    if step <= start_descent:
        l_rate = start_val
    elif step <= end_step:
        l_rate = start_val + (step-start_descent)*(end_val-start_val)/(end_step-start_descent)
    else:
        l_rate = end_val
    return l_rate

# Specify early stopping criterion
stopping_level = 0.0015
stopping_batch_size = 30

# Specify when to display loss, plot, log, and save
display_step = np.max([1, 250//train_batch_size])
plot_step = 4*display_step
save_step = plot_step
validation_step = 2*plot_step
impulse_step = 20*validation_step
#stopping_step = 10*validation_step
stopping_step = 8*validation_step
log_step = display_step
metadata_step = 100*display_step

# Specify when to start saving prediction plots/logs
#PLOT_START = validation_step
SAVE_PLOTS = True
SAVE_IMPULSES = False

# Specify when to start writing summaries to log file
LOG_START = validation_step

# Specify when to check for early stopping
STOPPING_START = 500
    


solved_data = data_count
solved_data = solved_data - (solved_data%5)

# Specify maximum number of data points to use
MAX_TO_USE = None

# Specify current data available
if MAX_TO_USE:
    source_count = np.min([solved_data, MAX_TO_USE])
else:
    source_count = solved_data


# Set model name for saving checkpoints
model_name = 'model'

# Specify whether to train and/or test
TEST = False
TRAIN = (not TEST)


# Specify where to save model checkpoints
if TRAIN:
    model_dir = './Model/'
elif TEST:
    model_dir = './Testing/Model/'

# Define directory containing previously trained model
prev_model_dir = './Model/'

# 5-fold cross-calidation
if TRAIN and (starting_epoch == 1):
    cross_validation = source_count//5
    overall = np.random.permutation(range(source_count))
    cv_indices = np.reshape(overall, [5,-1])
else:
    cv_indices = np.load(prev_model_dir + 'cv_indices.npy')

    
# Determine training/validation counts
train_count = cv_indices[0:4,:].size
validation_count = cv_indices[4,:].size

# Determine batch sizes
train_batches = train_count//train_batch_size
validation_batches = validation_count//validation_batch_size

# Determine total number of batches
total_batches = epochs*train_batches

# Transforms image accoring to symmetry group of the square: [ROTATION, FLIP]
#transformations = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]]
transformations = [[0,0]]

# Determine total number of batches
transformations_count = len(transformations)
batches_per_epoch = train_batches*transformations_count
total_batches = epochs*batches_per_epoch


# Specify Directories
if TRAIN:
    data_directory = './Data/'
    array_directory = './Arrays/'


# Specify subdirectories for storing model information
checkpoint_directory = model_dir + 'Checkpoints/'
checkpoint_name = model_name + '_'
save_directory = checkpoint_directory + checkpoint_name
prediction_dir = model_dir + 'predictions/'
prediction_subdir = prediction_dir + 'predictions_'
log_directory =  model_dir + 'logs/'
logdir = log_directory + 'training_'
impulse_dir = model_dir + 'Impulses/'
impulse_subdir = impulse_dir + 'impulses_'

# Specify checkpoint directory for previously trained model
prev_checkpoint_directory = prev_model_dir + 'Checkpoints/' + checkpoint_name 

# Specify if previous checkpoint should be loaded
if (TEST) or (starting_epoch > 1):
    LOAD_PREVIOUS = True
else:
    LOAD_PREVIOUS = False


# Run setup to initialize model directory
def run_setup(variable_count, NEW_CV=True):

    # Make Model Directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save cross-validation indices
    if NEW_CV:
        np.save(model_dir + 'cv_indices.npy', cv_indices)

    # Make backup copies of files used for model
    copyfile('Run.py', model_dir + 'Run.py')
    copyfile('model.py', model_dir + 'model.py')
    copyfile('reader.py', model_dir + 'reader.py')
    copyfile('parameters.py', model_dir + 'parameters.py')
    copyfile('convolution_layers.py', model_dir + 'convolution_layers.py')
    #copyfile('freeze.py', model_dir + 'freeze.py')
    
    # Make Log Directory
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        
    # Make Log Subdirectories
    for k in range(1, 5+1):
        if not os.path.exists(logdir + str(k)):
            os.makedirs(logdir + str(k))

    # Make Checkpoint Directory
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    # Make Checkpoint Subdirectories
    for k in range(1, 5+1):
        if not os.path.exists(save_directory + str(k)):
            os.makedirs(save_directory + str(k))

    if SAVE_PLOTS:
        # Make Prediction Directory
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        # Make Prediction Subdirectories
        for k in range(1, 5+1):
            if not os.path.exists(prediction_subdir + str(k)):
                os.makedirs(prediction_subdir + str(k))
                lammps_dir = prediction_subdir + str(k) + '/LAMMPS/'
                os.makedirs(lammps_dir)


    # Write model information to file
    with open(model_dir + 'Model_Info.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter="\t", quoting = csv.QUOTE_NONE)
        csvwriter.writerow(['Model Parameters: %d' %(variable_count)])
        csvwriter.writerow(['Batch Size: %d' %(train_batch_size)])
        csvwriter.writerow(['Learning Rate: %0.5f' %(learning_rate)])
        csvwriter.writerow(['Rate/Parameter:  %0.3E' %(learning_rate/variable_count)])
        csvwriter.writerow(['Latent Variables: %d' %(latent)])
        csvwriter.writerow(['Dropout Rate: %0.2f' %(DROP)])
        csvwriter.writerow(['Batch Norm: %s' %(BATCH_NORM)])


def get_cv_indices(NEW_CV=True):

    if NEW_CV:
        cv_indices = cv_indices
    else:
        cv_indices = np.load(prev_model_dir + 'cv_indices.npy')

    return cv_indices



###
#     MODIFICATIONS FOR TEST SETTING
###

if TEST:

    epochs = 1
    test_batch_size = 1

    display_step = 1
    plot_step = 1
    save_step = 200

    solved_file = './Testing/current_data_ID.csv'
    with open(solved_file, 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        IDs = next(csvreader)
        test_count = int(IDs[0]) + 1

    test_batches = test_count//test_batch_size
    batches_per_epoch = test_batches*transformations_count
    total_batches = epochs*batches_per_epoch
