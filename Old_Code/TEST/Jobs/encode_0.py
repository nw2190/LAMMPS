import os
from encoder import *
array_directory = "./Arrays/"
data_count = 1000
START = 0
END = 630
CONVERT_DUMP = True
if not os.path.exists(array_directory):
    os.makedirs(array_directory)
make_template()
for n in range(START,END):
    if CONVERT_DUMP:
        encode(n)
