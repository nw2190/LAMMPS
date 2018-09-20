import numpy as np
import pandas as pd
import time
import csv

#from parameters import *

data_directory = './Data/'
array_directory = './Arrays/'
resolution = 64
layers = 1

# Specify domain under consideration
tolerance = 0.0008
[x_min, x_max] = [-0.037 - tolerance, 0.037 + tolerance]
[y_min, y_max] = [-0.0025 - tolerance, 0.000 + tolerance]
[z_min, z_max] = [-0.037 - tolerance, 0.037 + tolerance]

# Specify the number of bins in each dimension
x_count = resolution
y_count = layers
z_count = resolution

# Define grids in each dimension
#x_grid = np.linspace(x_min,x_max,x_count + 1)
#y_grid = np.linspace(y_min,y_max,y_count + 1)
#z_grid = np.linspace(z_min,z_max,z_count + 1)


def encode(ID):
    ID = ID + 1
    # Specify whether or not to plot bin counts
    PLOT = False

    # Specify resolution
    #resolution = 64

    # Define number of layers in y-direction
    layers = 1
    
    # Define filenames
    input_filename = data_directory + 'dump_' + str(ID) +'.peri'
    output_count_filename =  array_directory + 'counts_' + str(ID) + '.npy'
    output_damage_filename =  array_directory + 'damage_' + str(ID) + '.npy'
    output_average_filename =  array_directory + 'avg_damage_' + str(ID) + '.npy'
    
    # Row numbers for relevant information
    header_length = 9
    time_row = 1
    atoms_row = 3
    header_row = 8
    
    # Specify number of relevant columns
    number_of_cols = 6
    
    # Determine the total number of atoms
    tmp_dataframe = pd.read_csv(input_filename, sep=' ', skiprows=atoms_row-1, nrows=1)
    atom_array = np.array(tmp_dataframe.as_matrix())
    atom_count = int(atom_array[0,0])
    #print('\nNumber of Atoms: %d' %(atom_count))


    skip_rows = header_length + int(atom_count) + header_length - 1
    # Read rows for final state of the system
    dataframe = pd.read_csv(input_filename, sep=' ', skiprows=skip_rows, nrows=atom_count)
    array = np.array(dataframe.as_matrix())
    array = array[:,0:number_of_cols]

    # ID_array stores [ID, Type]
    ID_array = array[:,0:2].astype(np.int8)

    # data_array stores [x, y, z, damage]
    data_array = array[:,2:number_of_cols].astype(np.float32)

    # Specify domain under consideration
    #[x_min, x_max] = [-0.037, 0.037]
    #[y_min, y_max] = [-0.0025, 0.000]
    #[z_min, z_max] = [-0.037, 0.037]

    
    # Specify the number of bins in each dimension
    #x_count = resolution
    #y_count = layers
    #z_count = resolution

    # Define grids in each dimension
    x_grid = np.linspace(x_min,x_max,x_count + 1)
    y_grid = np.linspace(y_min,y_max,y_count + 1)
    z_grid = np.linspace(z_min,z_max,z_count + 1)

    # Determine x-index of bin
    def get_x_bin(x):
        for n in range(0,x_count):
            if x_grid[n] <= x <= x_grid[n+1]:
                return n
            elif n==x_count-1:
                return -1

    # Determine y-index of bin
    def get_y_bin(y):
        for n in range(0,y_count):
            if y_grid[n] <= y <= y_grid[n+1]:
                return n
            elif n==y_count-1:
                return -1

    # Determine z-index of bin
    def get_z_bin(z):
        for n in range(0,z_count):
            if z_grid[n] <= z <= z_grid[n+1]:
                return n
            elif n==z_count-1:
                return -1


    total_count = 0
    bin_counts = np.zeros([x_count,z_count,y_count])
    bin_damage = np.zeros([x_count,z_count,y_count])

    # Sort each atom into the appropriate bin
    for n in range(0,atom_count):
        x_bin = get_x_bin(data_array[n,0])
        y_bin = get_y_bin(data_array[n,1])
        z_bin = get_z_bin(data_array[n,2])


        if (not z_bin == -1) and (not y_bin == -1) and (not x_bin == -1):
            bin_counts[x_bin,z_bin,y_bin] += 1
            bin_damage[x_bin,z_bin,y_bin] += data_array[n,3]
            total_count += 1
        #else:
        #    print('POINT NOT FOUND:  ( %f , %f , %f )' %(data_array[n,0],data_array[n,1],data_array[n,2]))


    avg_count = 0
    avg_damage = np.zeros([x_count,z_count,y_count])
    for i in range(0,x_count):
        for j in range(0,y_count):
            for k in range(0,z_count):
                if bin_counts[i,k,j] > 0:
                    av_d = bin_damage[i,k,j]/bin_counts[i,k,j]
                    if av_d == 0.0:
                        avg_damage[i,k,j] = 0.0001
                    else:
                        avg_damage[i,k,j] = av_d
                    avg_count += 1
                    
    
    # Save atom counts and damage for each bin
    np.save(output_count_filename, bin_counts)
    np.save(output_damage_filename, bin_damage)
    np.save(output_average_filename, avg_damage)

#    print('Non-zero entries for ID = ' + str(ID) + ':  ' + str(np.count_nonzero(avg_damage)))
#    print(avg_damage.shape)

    # Display the number of atoms found in domain
    print('[Dump File %d] Atoms Found: %d   Condensed Count: %d' %(ID,int(total_count), int(avg_count)))


    if PLOT:
        # Plot bin counts
        for i in range(0,y_count):
            plt.matshow(bin_counts[:,:,i],cmap='Greys')
        plt.show()


