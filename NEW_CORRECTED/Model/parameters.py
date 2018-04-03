### LIST OF MODEL PARAMETERS
import numpy as np
import csv
import os

NEW_CV_INDICES = True

# Set model name for saving checkpoints
model_name = 'model'

# Specify current meshes and data available
#data_count = 2325
data_count = 2920

if NEW_CV_INDICES:
    cross_validation = data_count//5
    overall = np.random.permutation(range(1,data_count+1))
    cv_indices = np.reshape(overall, [5,-1])
    np.save('cv_indices.npy',cv_indices)
else:
    cv_indices = np.load('cv_indices.npy')


train_indices = np.concatenate( (cv_indices[0:0], cv_indices[1:5,:]) ).flatten()
test_indices = cv_indices[0,:]

train_count = cv_indices[0:4,:].size

# Specify Training Resolution
train_resolution = 64
resolution = train_resolution

# Specify current epoch number and number of epochs to run
starting_epoch = 1
epochs = 1000

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
display_step = 100
plot_step = 100
save_step = 1000


# Specify whether to train and/or test
TEST = False
TRAIN = (not TEST)

# Specify if previous checkpoint should be loaded
LOAD_PREVIOUS = False

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
