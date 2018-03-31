import argparse
import numpy as np
import csv

from encoder import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", default=1, type=int, help="ID of solution to plot")

    args = parser.parse_args()
    ID = args.ID

    data_directory = '../Data/'
    input_filename = data_directory + 'indenter_' + str(ID) +'.txt'
    indenter_data = []
    with open(input_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            indenter_data.append(row[0])

    data = np.array(indenter_data, dtype=np.float32)

    # Only use x,y coordinates
    data = data[0:2]

    #print('\nPlotting Solution for:   x = %f  y = %f\n' %(data[1],data[0]))
    print('\nPlotting Solution for:   (x,y)  = %f %f\n' %(data[1],data[0]))


    soln = np.load('../Arrays/avg_damage_'+str(ID)+'.npy')
    np.save('y_true_out',soln)
