import os
import sys
import glob
from dump import dump
from ensight import ensight

from encoder import *

## Note:
# 'data_count' is specified in 'parameters.py'

ID = 0 
decode('y_out.npy', ID, True)
        
#p_name = lammps_dir + 'prediction_dump_' + str(ID) + '.peri'
p_name = 'prediction_dump_' + str(ID) + '.peri'
dp = dump(p_name);
dp.map(1,'id',2,'type',3,'x',4,'y',5,'z',6,'damage');
ep = ensight(dp);
#ep.one(lammps_dir + 'p_disk_' + str(ID),'damage','Damage')
ep.one('p_disk_' + str(ID),'damage','Damage')
