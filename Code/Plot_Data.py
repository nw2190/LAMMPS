import numpy as np
import matplotlib.pyplot as plt
import csv


data_directory = './Data/'
N_START = 0
#N = 2325
#N = 2925
N = 25250

        


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


indices = [n for n in range(0,N)]

count = 0
vals = np.zeros([len(indices),2])
for n in indices:
    try:
        val = read_data(N_START+n)
        vals[count,:] = val
    except Exception:
        pass
    count += 1

print(vals.shape)
marker_size = 5.0
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
