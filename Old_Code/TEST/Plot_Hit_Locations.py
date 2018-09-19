import numpy as np
import matplotlib.pyplot as plt
import csv


data_directory = './Data/'
N_START = 0
#N = 2325
N = 3150

        


# Read and Transform Indenter File
def read_data(ID, **keyword_parameters):
    #data_label = 'indenter_' + str(ID)
    #data_file = array_directory + data_label + '.npy'
    #vals = np.load(data_file)

    #template_file = array_directory + 'template.npy'
    #template = np.load(template_file)

    #SCALING = 10e3
    SCALING = 1
    
    # Define filenames
    input_filename = data_directory + 'indenter_' + str(ID) +'.txt'
    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])
            
    data = np.array(indenter_data, dtype=np.float32)

    # Only use x,y coordinates
    data = data[0:2]

    # Only use x coordinate
    #data = data[0:1]

    data = SCALING*data
    
    vals_array = np.array(data,dtype=np.float32)
    #print(vals_array)
    return vals_array




PLOT_CV = False
marker_size = 50.0

if PLOT_CV:
    cv_indices = np.load('./cv_indices.npy')
    train_indices = np.concatenate( (cv_indices[0:0], cv_indices[1:5,:]) ).flatten()
    test_indices = cv_indices[0,:]

    count = 0
    train_vals = np.zeros([train_indices.size,2])
    for n in train_indices:
        val = read_data(N_START+n+1)
        train_vals[count,:] = val
        count += 1

    print(train_vals.shape)
    
    plt.scatter(train_vals[:,0],train_vals[:,1], s=marker_size, c='b')
    
    
    count = 0
    test_vals = np.zeros([test_indices.size,2])
    for n in test_indices:
        val = read_data(N_START+n+1)
        test_vals[count,:] = val
        count += 1

    plt.scatter(test_vals[:,0],test_vals[:,1], s=marker_size, c='r')

else:
    count = 0
    vals = np.zeros([N,2])
    indices = [n for n in range(N_START,N_START+N)]
    for n in indices:
        val = read_data(N_START+n+1)
        vals[count,:] = val
        count += 1

    plt.scatter(vals[:,0],vals[:,1], s=marker_size, c='b')
    
plt.axis('equal')
scale = 0.0125
xmin = -scale
xmax = scale
ymin = -scale
ymax = scale
plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))


plt.show()
