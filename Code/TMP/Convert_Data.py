import os
#from parameters import *
from encoder import *

## Note:
# 'data_count' is specified in 'parameters.py'

START = 2324
END = 2325

CONVERT_DUMP = True
#CONVERT_INDENT = True

# Make Array Directory
if not os.path.exists(array_directory):
    os.makedirs(array_directory)



for n in range(START,END):
    if CONVERT_DUMP:
        encode(n+1)

    #if CONVERT_INDENT:
    #    indenter_array(n)
