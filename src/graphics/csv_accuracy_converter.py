import csv
import numpy as np
name = input('filename?')
mode = 'pgf'

with open(name, 'r') as file:
    reader = csv.reader(file)
    for _ in reader:
        length = len(_)
    
        
bins = np.zeros((21,length-1))
bins[0] = np.array([i for i in range(length-1)])

# t, b0, b1, ... b10, d0, d1, ... d10





with open(name, 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        x = []
        [x.append(float(t)) for t in row]
        x = np.array(x)
        diff = np.diff(x)
        for i in range(length-1):
            if x[i] > 0.9: 
                bins[1, i] += 1
                bins[11, i] += diff[i]
            elif x[i] > 0.8:
                bins[2, i] += 1
                bins[12, i] += diff[i]
            elif x[i] > 0.7:
                bins[3, i] += 1
                bins[13, i] += diff[i]
            elif x[i] > 0.6:
                bins[4, i] += 1
                bins[14, i] += diff[i]
            elif x[i] > 0.5:
                bins[5, i] += 1
                bins[15, i] += diff[i]
            elif x[i] > 0.4:
                bins[6, i] += 1
                bins[16, i] += diff[i]
            elif x[i] > 0.3:
                bins[7, i] += 1
                bins[17, i] += diff[i]
            elif x[i] > 0.2:
                bins[8, i] += 1
                bins[18, i] += diff[i]
            elif x[i] > 0.1:
                bins[9, i] += 1
                bins[19, i] += diff[i]
            else:
                bins[10, i] += 1
                bins[20, i] += diff[i]
    
    
    for i in range(10):
        bins[11+i,bins[1+i,:]>0] = bins[11+i,bins[1+i,:]>0] / bins[1+i,bins[1+i,:]>0]

if mode == 'csv':
    with open(name.split('.')[0]+'_plot.csv', 'w') as file:
        file.write('t,b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9\n')
        
        for i in range(bins.shape[1]):
            for j in range(bins.shape[0]):
                if j < bins.shape[0] - 1:
                    file.write(str(bins[j,i])+',')
                else:
                    file.write(str(bins[j,i])+'\n')
if mode == 'pgf':
    with open(name.split('.')[0] + '_N' + '.csv','w+') as f:
        np.savetxt(f, bins[1:11,:].astype('i'), delimiter=",", fmt='%i')
    with open(name.split('.')[0] + '_phi' + '.csv','w+') as f:
        np.savetxt(f, np.arctan(bins[11:21,:]*10)*180/np.pi, delimiter=",", fmt='%2.6f')
