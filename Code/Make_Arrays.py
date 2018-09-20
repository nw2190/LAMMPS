from __future__ import division
import multiprocessing
import sys
import os

from encoder import *

if __name__ == '__main__':

    # Specify total data count
    data_count = 25250

    # Only make template
    TEMPLATE_ONLY = False

    # Check that array directory exists
    array_directory = "./Arrays/"
    if not os.path.exists(array_directory):
        os.makedirs(array_directory)

    # Create template array
    make_template()


    # Function for multiprocessing pool
    def convert_arrays(d):
        for n in range(d[0],d[1]):
            if not TEMPLATE_ONLY:
                encode(n, verbose=False)
        
    # Create multiprocessing pool
    NumProcesses = 4
    pool = multiprocessing.Pool(processes=NumProcesses)


    # Define starting and ending indices for loops
    start_indices = [2500*n  for  n in range(0,10+1)]
    end_indices = [2500*n  for  n in range(1,11+1)]
    end_indices[-1] = data_count

    print('\n [ Converting Dump Files to Arrays ]\n')
    num_tasks = len(start_indices)
    for i, _ in enumerate(pool.imap_unordered(convert_arrays, [d for d in zip(start_indices, end_indices)]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')
    




