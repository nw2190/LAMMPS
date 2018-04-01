import csv

N = 5

for n in range(0,N):

    python_code = 'python encode_' + str(n) + '.py'
    header = ['#!/bin/sh -l',
              '# FILENAME:  myjobsubmissionfile',
              'module load python',
              '# Change to the directory from which you originally submitted this job.',
              'cd $PBS_O_WORKDIR',
              python_code]
    
    output_filename = 'job' + str(n) 
    with open(output_filename, 'w') as csvfile:
        for line in header:
            csvfile.write(line+'\n')
