import os
import sys
import glob
from dump import dump
from ensight import ensight

from parameters import *
from encoder import *

## Note:
# 'data_count' is specified in 'parameters.py'

# Make LAMMPS Directory
if not os.path.exists(lammps_dir):
    os.makedirs(lammps_dir)
else:
    print('\nWarning: LAMMPS directory already exists.\n')

skip_IDs = []

for ID in range(1, data_count + 1):
    p_files = glob.glob(prediction_dir + '*_prediction_' + str(ID) + '.npy')
    s_files = glob.glob(prediction_dir + '*_solution_' + str(ID) + '.npy')

    if not p_files:
        # list is empty
        skip_IDs.append(ID)
    else:
        p_file = p_files[-1]
        s_file = s_files[-1]

        decode(p_file, ID, True)
        decode(s_file, ID, False)
    

# Change to LAMMPS directory to store dump files
os.chdir(lammps_dir)

for ID in range(1, data_count + 1):
    if not (ID in skip_IDs):
        
        #p_name = lammps_dir + 'prediction_dump_' + str(ID) + '.peri'
        p_name = 'prediction_dump_' + str(ID) + '.peri'
        dp = dump(p_name);
        dp.map(1,'id',2,'type',3,'x',4,'y',5,'z',6,'damage');
        ep = ensight(dp);
        #ep.one(lammps_dir + 'p_disk_' + str(ID),'damage','Damage')
        ep.one('p_disk_' + str(ID),'damage','Damage')
        
        #s_name = lammps_dir + 'solution_dump_' + str(ID) + '.peri'
        s_name = 'solution_dump_' + str(ID) + '.peri'
        ds = dump(s_name);
        ds.map(1,'id',2,'type',3,'x',4,'y',5,'z',6,'damage');
        es = ensight(ds);
        #es.one(lammps_dir + 's_disk_' + str(ID),'damage','Damage')
        es.one('s_disk_' + str(ID),'damage','Damage')
