import os
from encoder import *
array_directory = "./Arrays/"
data_count = 1000
START = 630
END = 1260
CONVERT_DUMP = True
if not os.path.exists(array_directory):
    os.makedirs(array_directory)
make_template()
for n in range(START,END):
    if CONVERT_DUMP:
        encode(n)