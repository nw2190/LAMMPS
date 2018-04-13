import csv


filename = './in.peri'

lines = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='&', quotechar='|')
    count = 1
    for row in csvreader:
        print(row)
        if row:
            lines.append(str(count) + ': ' + row[0] + '\n')
            count += 1
        #else:
            #lines.append('\n')
            



outfile = 'peri_file.txt'
with open(outfile, 'w') as csvfile:
    #csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='', quoting=csv.)

    for line in lines:
        #print(line)
        #csvwriter.writerow(line)
        csvfile.write(line)
