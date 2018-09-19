import os
import sys
import glob
from dump import dump
from ensight import ensight

from encoder import *

#COUNT = 200000
START = 100000
COUNT = 20000

pred_dir = './Model/results/'
lammps_dir = pred_dir + 'LAMMPS/'

if not os.path.exists(lammps_dir):
        os.makedirs(lammps_dir)


skip_IDs = []

for step in range(START, START + COUNT + 1):
    p_files = glob.glob(pred_dir + str(step) + '_prediction.npy')
    s_files = glob.glob(pred_dir + str(step) + '_solution.npy')

    if not p_files:
        # list is empty
        skip_IDs.append(step)
    else:
        p_file = p_files[-1]
        s_file = s_files[-1]

        decode(p_file, step, True, lammps_dir)
        decode(s_file, step, False, lammps_dir)
    

# Change to LAMMPS directory to store dump files
os.chdir(lammps_dir)

for ID in range(START, START + COUNT + 1):
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
