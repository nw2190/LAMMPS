import os
import csv
import numpy as np

DATA_COUNT = 2925

data_X = []
print('\nLoading Data...')
for n in range(0,DATA_COUNT):
    with open('./Data/indenter_' + str(n+1) + '.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        indenter_data = []
        for row in reader:
            indenter_data.append(row[0])
    data = np.array(indenter_data, dtype=np.float32)
    vals = data[0:2]
    data_X.append(vals)
np.save('DATA.npy',data_X)
    
data_y = []    
print('\nLoading Targets...')
for n in range(0,DATA_COUNT):
    vals = np.load('./Arrays/avg_damage_' + str(n+1) + '.npy')
    data_y.append(vals)
np.save('TARGETS.npy',data_y)
