import csv

N_START = 5
START = 1000
N = 5
increment = 200

for n in range(0,N):

    start = START + n*increment
    start_line = 'START = ' + str(start)
    end = START + n*increment + increment
    end_line = 'END = ' + str(end)

    header = ['import os',
              'from encoder import *',
              'array_directory = "./Arrays/"',
              'data_count = 1000',
              start_line,
              end_line,
              'CONVERT_DUMP = True',
              'if not os.path.exists(array_directory):',
              '    os.makedirs(array_directory)',
              'make_template()',
              'for n in range(START,END):',
              '    if CONVERT_DUMP:',
              '        encode(n)']
              

    
    output_filename = 'encode_' + str(N_START + n) + '.py'
    with open(output_filename, 'w') as csvfile:
        for line in header:
            csvfile.write(line+'\n')
