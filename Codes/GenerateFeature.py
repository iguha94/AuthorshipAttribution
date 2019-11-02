import os
import pandas as pd
import numpy as np
import csv

basedirectory='../BlogsAll/'
wtr = csv.writer(open ('AllFeatures.csv', 'w'), delimiter=',', lineterminator='\n')

directories=os.listdir(basedirectory)
totalcount=1
flag=0
for directory in directories:
    print('Processing Dir: ',directory)
    curdirectory = basedirectory+directory
    files = os.listdir(curdirectory)
    for file in files:
        if file.endswith(".pkl"):
            featurevector=[]
            columnvector=[]
            pickle_data = pd.read_pickle(curdirectory+'/'+file)
            for i in range(len(pickle_data)):
                featurevector.append(pickle_data[i][1])
                if flag==0:
                    columnvector.append(pickle_data[i][0])
            if flag==0:
                columnvector.append('Author_Id')
                wtr.writerow(columnvector)
                flag=1
            featurevector.append(int(directory))
            wtr.writerow(featurevector)
    totalcount=totalcount+1