import numpy as np
import pandas as pd
import time
import csv

from parameters import *

layers = output_layers

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


# Read and Transform Indenter File
def read_data(ID, **keyword_parameters):
    #data_label = 'indenter_' + str(ID)
    #data_file = array_directory + data_label + '.npy'
    #vals = np.load(data_file)

    #template_file = array_directory + 'template.npy'
    #template = np.load(template_file)

    SCALING = 10e3
    
    # Define filenames
    input_filename = data_directory + 'indenter_' + str(ID) +'.txt'
    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])
            
    data = np.array(indenter_data, dtype=np.float32)

    # Only use x,y coordinates
    #data = data[0:2]

    # Only use x coordinate
    data = data[0:1]

    data = SCALING*data
    
    #if ('transformation' in keyword_parameters):
    #    [R, F] = keyword_parameters['transformation']
    #    vals = np.rot90(vals, k=R)
    #    template = np.rot90(template, k=R)
    #    if F == 1:
    #        vals = np.flipud(vals)
    #        template = np.flipud(template)

    
    #vals_array = np.array([vals,template[:,:,0], template[:,:,1], template[:,:,2]])
    #vals_array = np.array([vals,template[:,:,0]])
    vals_array = np.array(data)
    #print(vals_array)
    return vals_array

# Read and Transform Indenter File
def read_soln(ID, **keyword_parameters):
    soln_label = 'avg_damage_' + str(ID)
    soln_file = array_directory + soln_label + '.npy'
    vals = np.load(soln_file)

    #template_file = array_directory + 'template.npy'
    #template = np.load(template_file)

    #print('Non-zero entries for ID = ' + str(ID) + ':  ' + str(np.count_nonzero(vals)))
    #print(vals.shape)

    layers = vals.shape[2]

    if ('transformation' in keyword_parameters):
        [R, F] = keyword_parameters['transformation']
        for n in range(0,layers):
            vals[:,:,n] = np.rot90(vals[:,:,n], k=R)
            #template = np.rot90(template, k=R)
        if F == 1:
            for n in range(0,layers):
                vals[:,:,n] = np.flipud(vals[:,:,n])
            #template = np.flipud(template)
    
    #vals_array = vals
    vals_array = np.array([vals[:,:,0]])  
    #vals_array = np.array([vals[:,:,0],template[:,:,0]])
    return vals_array




# Assemble training data using functions defined in 'reader.py'
# with image data returned as numpy arrays [vals, alpha].
def compile_data(ID, transform):
    # Recover data values from array
    x_data = read_data(ID, transformation = transform)
    
    # Recover solution values from array
    y_data = read_soln(ID, transformation = transform)
    y_data = np.min([y_data, np.ones_like(y_data)], 0)
    
    # Convert to NHWC format
    y_data = np.transpose(y_data, (1, 2, 0))
    return [x_data, y_data]


# Compiles batch of input/output data
def train_next_batch(M, data_indices, d_batch_size, transform):
    data = data_indices[M*d_batch_size:(M+1)*d_batch_size]

    [batch_x, batch_y] = [[],[]]
    for ID in data:
        [x_data, y_data] = compile_data(ID, transform)
        batch_x.append(x_data)
        batch_y.append(y_data)
        
    return [batch_x, batch_y]




def encode(ID):
    ID = ID + 1
    # Specify whether or not to plot bin counts
    PLOT = False

    # Specify resolution
    #resolution = 64

    # Define number of layers in y-direction
    layers = output_layers
    
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



def sharpen(val):
    gamma = 2.0
    exponent = -gamma*np.tan(np.pi*(val-0.5))
    sharp_val = 1.0/(1.0+np.exp(exponent))
    return sharp_val
    
def decode(filename, ID, PREDICTION, lammps_dir=lammps_dir):
    # Specify resolution
    #resolution = 64

    # Define number of layers in y-direction
    layers = output_layers

    # Define filenames
    #input_count_filename =  prediction_directory + 'counts_' + str(ID) + '.npy'
    #input_damage_filename =  prediction_directory + 'damage_' + str(ID) + '.npy'
    #input_average_filename =  prediction_directory + 'avg_damage_' + str(ID) + '.npy'
    input_average_filename = filename

    if PREDICTION:
        output_filename = lammps_dir + 'prediction_dump_' + str(ID) +'.peri'
    else:
        output_filename = lammps_dir + 'solution_dump_' + str(ID) +'.peri'

    #bin_counts = np.load(input_count_filename)
    #bin_damage = np.load(input_damage_filename)
    avg_damage = np.load(input_average_filename)
    
    # Specify domain under consideration
    #[x_min, x_max] = [-0.037, 0.037]
    #[y_min, y_max] = [-0.0025, 0.000]
    #[z_min, z_max] = [-0.037, 0.037]

    
    # Specify the number of bins in each dimension
    #x_count = resolution
    #y_count = 2*layers - 1
    #y_count = layers
    #z_count = resolution

    # Define grids in each dimension
    x_grid = np.linspace(x_min,x_max,x_count + 1)
    y_grid = np.linspace(y_min,y_max,y_count + 1)
    z_grid = np.linspace(z_min,z_max,z_count + 1)

    x_step = x_grid[1] - x_grid[0]
    y_step = y_grid[1] - y_grid[0]
    z_step = z_grid[1] - z_grid[0]

    x_grid = x_grid + 0.5*x_step
    y_grid = y_grid + 0.5*y_step
    z_grid = z_grid + 0.5*z_step

    count = 0
    line_list = []
    for i in range(0,x_count):
        for j in range(0,y_count):
            for k in range(0,z_count):
                damage = avg_damage[i,k,j]
                x = x_grid[i]
                y = y_grid[j]
                z = z_grid[k]
                if damage > 0.0:
                    if damage == 1.0:
                        sharp_damage = damage
                    else:
                        sharp_damage = sharpen(damage)
                    line = str(count) + ' 1 ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(sharp_damage)
                    line_list.append(line)
                    count += 1
                #count = bin_counts[i,k,j]
                #count_int = int(count)
                
                #if count_int > 0:
                #    damage = bin_damage[i,k,j]
                #    x = x_grid[i]
                #    y = y_grid[j]
                #    z = z_grid[k]
                #    atom_damage = damage/count
                    
                #    for n in range(0,count_int):
                #        line = str(ID) + ' 1 ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(atom_damage)
                #        line_list.append(line)
                #        ID += 1
                
    header = ['ITEM: TIMESTEP',
              '0',
              'ITEM: NUMBER OF ATOMS',
              str(count),
              'ITEM: BOX BOUNDS ss ss ss',
              '-3.7007399999999996e-02 3.7007399999999996e-02',
              '-2.0002499999999999e-03 2.5000000000000004e-07',
              '-3.7007399999999996e-02 3.7007399999999996e-02',
              'ITEM: ATOMS id type x y z c_1']
    with open(output_filename, 'w') as csvfile:
        #csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='', quoting=csv.QUOTE_MINIMAL)
        #csvwriter = csv.writer(csvfile, delimiter=' ',escapechar='', quoting=csv.QUOTE_NONE)
        for line in header:
            csvfile.write(line+'\n')

        for line in line_list:
            csvfile.write(line+'\n')










def indenter_array(ID):
    ID = ID + 1

    # Specify whether or not to plot bin counts
    PLOT = False

    # Specify resolution
    #resolution = 64

    # Specify domain under consideration
    #[x_min, x_max] = [-0.0375, 0.0375]
    #[y_min, y_max] = [-0.0025, 0.0005]
    #[z_min, z_max] = [-0.0375, 0.0375]

    # Define filenames
    input_filename = data_directory + 'indenter_' + str(ID) +'.txt'
    output_filename =  array_directory + 'indenter_' + str(ID) + '.npy'

    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])
            
    data = np.array(indenter_data, dtype=np.float32)
    
    x = data[0]
    z = data[1]

    if data.size == 2:
        r = 0.005
        v = 1.0
    elif data.size == 3:
        r = data[2]
        v = 1.0
    else:
        r = data[2]
        v = data[3]

    array = np.zeros((resolution,resolution))

    # Define grids in each dimension
    x_grid = np.linspace(x_min,x_max,resolution + 1)
    z_grid = np.linspace(z_min,z_max,resolution + 1)
    x_step = x_grid[1] - x_grid[0]
    z_step = z_grid[1] - z_grid[0]
    x_grid = x_grid + x_step
    z_grid = z_grid + z_step

    epsilon = r/3
    
    for i in range(0,resolution):
        for j in range(0,resolution):
            x_i = x_grid[i]
            z_j = z_grid[j]
            dist = np.sqrt(np.power(x-x_i,2) + np.power(z-z_j,2))
            if dist <= r + epsilon:
                # Use Image Format
                #array[resolution - j - 1,i] = v
                # Use Array Format
                array[i,j] = v

    np.save(output_filename, array)

    if PLOT:
        plt.matshow(array,cmap='Greys')
        plt.show()








        

def make_template():

    PLOT = False
    
    # Specify resolution
    #resolution = 64

    # Define number of layers in y-direction
    layers = output_layers
    
    # Define filenames
    input_filename = data_directory + 'dump_' + str(1) +'.peri'
    output_filename =  array_directory + 'template' + '.npy'
    
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


    #skip_rows = header_length + int(atom_count) + header_length - 1
    skip_rows = header_length  - 1
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
    #y_count = 2*layers - 1
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
                    avg_damage[i,k,j] = 1.0
                    avg_count += 1
                    
    
    # Save atom counts and damage for each bin
    np.save(output_filename, avg_damage)

#    print('Non-zero entries for ID = ' + str(ID) + ':  ' + str(np.count_nonzero(avg_damage)))
#    print(avg_damage.shape)

    # Display the number of atoms found in domain
    print('Template Atoms Found: %d   Condensed Count: %d' %(int(total_count), int(avg_count)))


    if PLOT:
        # Plot bin counts
        for i in range(0,y_count):
            plt.matshow(avg_damage[:,:,i],cmap='Greys')
        plt.show()



def display_progress(loss, start_time, step, total_batches, epoch):
    
    
    current_time = time.clock()
    time_elapsed = current_time - start_time
    rate = time_elapsed/step
    approx_finish = rate * (total_batches - step)
    hours = np.floor(approx_finish/3600.0)
    if hours > 0:
        minutes = np.floor((approx_finish/3600.0 - hours) * 60)
        seconds = np.floor(((approx_finish/3600.0 - hours) * 60 - minutes) * 60)
        progress = '   [ Estimated Time  ~  ' + str(int(hours)) + 'h  %2sm  %2ss ]'   %(str(int(minutes)),str(int(seconds)))
    else:
        minutes = np.floor(approx_finish/60.0)
        seconds = np.floor((approx_finish/60.0 - minutes) * 60)
        progress = '   [ Estimated Time  ~  ' + str(int(minutes)) + 'm  ' + str(int(seconds)) + 's ]'
        #" of " + str(epochs) + \
    print("Epoch " + str(epoch+1) + \
          " - Iter " + str(step) + ' of ' + str(total_batches) + \
          ":   Minibatch Loss= " + "{:.6f}".format(loss)  + progress)
