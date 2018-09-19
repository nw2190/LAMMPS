import os
from encoder import *

array_directory = './Arrays/'

# Specify current data available
data_count = 40000

START = 0
END = data_count

CONVERT_DUMP = True

# Make Array Directory
if not os.path.exists(array_directory):
    os.makedirs(array_directory)

make_template()

for n in range(START,END):
    if CONVERT_DUMP:
        encode(n)

