import numpy as np
import matplotlib.pyplot as plt
import csv


data_directory = './Data/'
N_START = 0
N = 2325

        


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
    data = data[0:2]

    # Only use x coordinate
    #data = data[0:1]

    data = SCALING*data
    
    vals_array = np.array(data)
    #print(vals_array)
    return vals_array


vals = np.zeros([N,2])

for n in range(0,N):
    val = read_data(N_START+n+1)
    vals[n-1,:] = val

plt.scatter(vals[:,0],vals[:,1])
plt.show()
