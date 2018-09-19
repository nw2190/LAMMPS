import os
import numpy as np
import csv

DATA_COUNT = 500

    
data_X = []    
print('\nLoading Data...')
for n in range(1,DATA_COUNT+1):
    input_filename = './Data/indenter_' + str(n) +'.txt'
    SCALING = 10e3
    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])

    vals = np.array(indenter_data, dtype=np.float32)

    # Only use x,y coordinates
    #data = data[0:2]

    # Only use x coordinate
    vals = vals[0:1]

    vals = SCALING*vals
    data_X.append(vals)
np.save('DATA.npy',data_X)


# load data
print('\nLoading Targets...')
data_y = []
for n in range(1,DATA_COUNT+1):
    vals = np.load('./Arrays/avg_damage_' + str(n) + '.npy')
    data_y.append(vals)
np.save('TARGETS.npy',data_y)
