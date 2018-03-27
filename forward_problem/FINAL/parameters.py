### LIST OF MODEL PARAMETERS
import numpy as np
import csv
import os

# Set model name for saving checkpoints
model_name = 'model'

# Specify current meshes and data available
data_count = 1000

# Specify Training Resolution
train_resolution = 64
resolution = train_resolution

# Specify current epoch number and number of epochs to run
starting_epoch = 1
epochs = 0

# Training Parameters
learning_rate = 0.0001

# Specify batch sizes
data_batch_size = 1
batch_size = data_batch_size

# Specify Output Layers
output_layers = 1

# Problem Setup Parameters
n_channels_in = 2     ## Number of channels for input images 
n_channels_out = 1    ## Number of channels for output images 

# Specify Intermediate Channel Sizes and Node Counts
#nodes = [16, 32, 16*16*resolution]
#middle_channels = 64
#middle_res = 8
#nodes = [25, 50, middle_res*middle_res*middle_channels]
#channels = [64, 32, 16, n_channels_out]

# Specify Kernel/Filter Sizes
#kernels = [3, 3, 3]

# Specify when to display loss, plot, and save
#display_step = 25
#plot_step = 50
#save_step = 100
display_step = 10
plot_step = 10
save_step = 10


# Specify whether to train and/or test
TEST = False
TRAIN = (not TEST)

# Specify if previous checkpoint should be loaded
LOAD_PREVIOUS = True

# Specify Directories
data_directory = './Data/'
array_directory = './Arrays/'

# Specify where to save model checkpoints
model_dir = './Model/'
checkpoint_directory = model_dir + 'Checkpoints/'
checkpoint_name = 'epoch' + str(starting_epoch) + '_' + model_name
save_file = checkpoint_directory + checkpoint_name

# Specify where to store predictions and LAMMPS files
prediction_dir = model_dir + 'predictions/'
lammps_dir = prediction_dir + 'LAMMPS/'

log_directory =  model_dir + 'logs/'
logdir = log_directory + 'epoch' + str(starting_epoch) + '/'




# Specify where to save predictions during testing
testing_prediction_dir = './Testing/predictions/'

# Specify when to display loss / plot when testing
test_display_step = 1
test_plot_step = 1

# Define weights for cost function
cost_weights = {'ms' : 1.0, 'bdry' : 0.5}


BATCH_NORM = False

# Transforms image accoring to symmetry group of the square: [ROTATION, FLIP]
#transformations = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]]
transformations = [[0,0]]


# Make Model Directory
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Make Log Directory
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Make Log Subdirectory
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Make Checkpoint Directory
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)
    if LOAD_PREVIOUS:
        print('Warning: There are no checkpoints to load.')

# Make Prediction Directory
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


# Determine total number of batches
transformations_array = np.array(transformations)
transformations_count = (transformations_array.shape)[0]
