from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

import glob
import sys
import csv
import time
import string
import heapq

SCALING = 100.0


# Recover values and alpha mask corresponding to data image
def read_data(data_ID, data_directory, **keyword_parameters):
    data_label = 'data_' + str(data_ID)
    data_file = data_directory + data_label + '.npy'
    vals = np.load(data_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)

    vals_array = np.array([vals])
    return vals_array



# Recover values and alpha mask corresponding to solution image 
def read_soln(source_ID, solution_directory, **keyword_parameters):
    ID_label = '0_' + str(source_ID)
    soln_label = 'solution_' + ID_label
    soln_file = solution_directory + soln_label + '.npy'
    vals = np.load(soln_file)

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        vals = np.rot90(vals, k=R)
        if F == 1:
            vals = np.flipud(vals)
    
    vals = SCALING*vals
    vals_array = np.array([vals])
    return vals_array


# Assemble training data using functions defined in 'reader.py'
def compile_data(k, data_dir, solution_dir, transform):
    
    # Read data files
    source = read_data(k, data_dir, transformation=transform)
    x_data = np.array(source)

    # Read solution file
    solution = read_soln(k, solution_dir, transformation=transform)
    y_data = np.array(solution)

    # Convert to NHWC format
    x_data = np.transpose(x_data, (1, 2, 0))
    y_data = np.transpose(y_data, (1, 2, 0))
    return [x_data, y_data]


# Compile single batch of input/output data
def train_next_batch(K, batch_size, indices, transformation=[0,0]):
    source = indices[K*batch_size:(K+1)*batch_size]
    [batch_x, batch_y] = [[],[]]
    for k in source:
        [x_data, y_data] = compile_data(k, data_directory, solution_directory, transformation)
        batch_x.append(x_data)
        batch_y.append(y_data)
    return batch_x, batch_y



def get_time():
    return time.time()

def convert_time(t):
    hours = np.floor(t/3600.0)
    minutes = np.floor((t/3600.0 - hours) * 60)
    seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
    if hours > 0:
        minutes = np.floor((t/3600.0 - hours) * 60)
        seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
        t_str = str(int(hours)) + 'h  ' + \
                str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    elif (hours == 0) and (minutes >= 1):
        minutes = np.floor(t/60.0)
        seconds = np.ceil((t/60.0 - minutes) * 60)
        t_str = str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    else:
        seconds = (t/60.0 - minutes) * 60
        t_str = str(seconds) + 's'

        
    return t_str







# Apply median filter to two-dimensional array
def median_filter(vals):
    resolution = vals.shape[0]
    padded = np.lib.pad(vals, (1,), 'constant', constant_values=(0.0,0.0))
    
    for i in range(1,resolution+1):
        for j in range(1,resolution+1):
            vals[i-1,j-1] = np.median(padded[i-1:i+2,j-1:j+2])

    return vals

# Apply mean filter to two-dimensional array
def mean_filter(vals):
    resolution = vals.shape[0]
    padded = np.lib.pad(vals, (1,), 'constant', constant_values=(0.0,0.0))
    
    for i in range(1,resolution+1):
        for j in range(1,resolution+1):
            vals[i-1,j-1] = np.mean(padded[i-1:i+2,j-1:j+2])

    return vals



# Plots predictions with matplotlib
def plot_prediction(ID, vals, Model=0, CV=1, Rm_Outliers=False, Filter=True, Plot_Error=True):

    mpl.style.use('classic')
    
    soln_file = '../Setup/Solutions/solution_0_' + str(ID) + '.npy'
        
    # Retrieve solution
    soln = SCALING*np.load(soln_file)

    # Load network prediction
    pred = vals[0,:,:,0]

    pred_vals, plot_X, plot_Y = preprocesser(pred, Rm_Outliers=False, Filter=True)
    soln_vals, plot_X, plot_Y = preprocesser(soln, Rm_Outliers=False, Filter=False)

    # Determine solution/prediction extrema
    soln_min = np.min(soln_vals)
    soln_max = np.max(soln_vals)
    pred_min = np.min(pred_vals)
    pred_max = np.max(pred_vals)

    z_min = np.min([pred_min, soln_min])
    z_max = np.max([pred_max, soln_max])
    epsilon = 0.05*(z_max - z_min)

    def l2_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sqrt(np.sum(np.power(y_ - y, 2)))

    def rel_l2_error(y_,y):
        return np.sqrt(np.sum(np.power(y_ - y, 2)))/np.sqrt(np.sum(np.power(y, 2)))

    def l1_error(y_,y):
        resolution = y.shape[0]
        scaling = np.power(1.0/(resolution - 1.0),2)
        return scaling*np.sum(np.abs(y_ - y))

    def rel_l1_error(y_,y):
        return np.sum(np.abs(y_ - y))/np.sum(np.abs(y))

    l2_e = l2_error(pred_vals, soln_vals)
    rl2_e = rel_l2_error(pred_vals, soln_vals)
    l1_e = l1_error(pred_vals, soln_vals)
    rl1_e = rel_l1_error(pred_vals, soln_vals)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, pred_vals, cmap='hot')
    pred_title = 'Prediction:    min = %.6f    max = %.6f'  %(pred_min, pred_max)
    ax1.set_title(pred_title)
    ax1.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(plot_X,plot_Y, soln_vals, cmap='hot')
    soln_title = 'Solution:    min = %.6f    max = %.6f'  %(soln_min, soln_max)
    ax2.set_title(soln_title)
    ax2.set_zlim([z_min - epsilon, z_max + epsilon])
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')

    fig.suptitle('L^1: %.5f        L^2: %.5f\nL^1 Rel: %.5f     L^2 Rel: %.5f' %(l1_e,l2_e,rl1_e,rl2_e), fontsize=24)


    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    if Plot_Error:
        diff = soln_vals - pred_vals
        diff_min = np.min(diff)
        diff_max = np.max(diff)
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(plot_X,plot_Y,diff, cmap='hot')
        ax.set_title('Error Min: %.6f          Error Max:  %.6f' %(diff_min,diff_max))
        #ax.set_title('L^1: %.5f , L^1 Rel: %.5f , L^2: %.5f , L^2 Rel: %.5f' %(l1_e,rl1_e,l2_e,rl2_e))
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')

        # SYSTEM 1
        #mng = plt.get_current_fig_manager()
        #mng.frame.Maximize(True)
        
        # SYSTEM 2
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        
        # SYSTEM 3
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        
    # Bind axes for comparison
    def on_move(event):
        if event.inaxes == ax1:
            if ax1.button_pressed in ax1._rotate_btn:
                ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            elif ax1.button_pressed in ax1._zoom_btn:
                ax2.set_xlim3d(ax1.get_xlim3d())
                ax2.set_ylim3d(ax1.get_ylim3d())
                ax2.set_zlim3d(ax1.get_zlim3d())
        elif event.inaxes == ax2:
            if ax2.button_pressed in ax2._rotate_btn:
                ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            elif ax2.button_pressed in ax2._zoom_btn:
                ax1.set_xlim3d(ax2.get_xlim3d())
                ax1.set_ylim3d(ax2.get_ylim3d())
                ax1.set_zlim3d(ax2.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
                
    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()




# Plots data functions with matplotlib
def plot_data(source_ID):
    data_file = './Setup/Data/data_' + str(source_ID) + '.npy'
        
    # Load data function
    data = np.load(data_file)
    resolution = data.shape[0]
    
    data_vals, plot_X, plot_Y = preprocesser(data, Rm_Outliers=False, Filter=False)
    data_min = np.min(data_vals)
    data_max = np.max(data_vals)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, data_vals, cmap='hot')
    data_title = 'Data:    min = %.6f    max = %.6f'  %(data_min, data_max)
    ax1.set_title(data_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
        
    plt.show()



# Plots data functions with matplotlib
def plot_soln(soln_ID):
    data_file = './Setup/Solutions/solution_' + str(0) + '_' + str(soln_ID) + '.npy'
        
    # Load data function
    data = SCALING*np.load(data_file)
    resolution = data.shape[0]

    data_vals, plot_X, plot_Y = preprocesser(data, Rm_Outliers=False, Filter=False)
    data_min = np.min(data_vals)
    data_max = np.max(data_vals)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(plot_X,plot_Y, data_vals, cmap='hot')
    data_title = 'Data:    min = %.6f    max = %.6f'  %(data_min, data_max)
    ax1.set_title(data_title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
        
    plt.show()




# Plots predictions with matplotlib
def preprocesser(vals, refine=2, Rm_Outliers=False, Filter=True, Median=False, Mean=True):

    # Determine spatial resolution
    resolution = vals.shape[0]
    
    
    if Rm_Outliers:
        # Identify and remove outliers
        outlier_buffer = 5
        
        vals_list = vals.reshape((resolution*resolution,))
        vals_mins = heapq.nsmallest(outlier_buffer, vals_list)
        vals_maxes = heapq.nlargest(outlier_buffer, vals_list)

        # Cap max and min
        vals_min = np.max(vals_mins)
        vals_max = np.min(vals_maxes)
        
        # Trim outliers
        over  = (vals > vals_max)
        under = (vals < vals_min)

        # Remove outliers
        vals[over] = vals_max
        vals[under] = vals_min
        
    else:
        vals_min = np.max(vals)
        vals_max = np.min(vals)

    if Filter:
        # Apply median/mean filter
        if Median:
            vals = median_filter(vals)
        if Mean:
            vals = mean_filter(vals)

    # Create grid
    start = 0.0
    end = 1.0
    x = np.linspace(start,end,resolution)
    y = np.linspace(start,end,resolution)

    [X, Y] = np.meshgrid(x,y)

    interp_vals = interp2d(x,y, vals, kind='cubic')

    # Create refined grid
    plot_start = 0.0
    plot_end = 1.0
    
    plot_x = np.linspace(plot_start,plot_end,refine*resolution)
    plot_y = np.linspace(plot_start,plot_end,refine*resolution)

    [plot_X, plot_Y] = np.meshgrid(plot_x, plot_y)

    vals_int_values = interp_vals(plot_x, plot_y)

    return vals_int_values, plot_X, plot_Y
