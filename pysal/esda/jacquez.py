"""
An implementation of Jacquez' k-nearest neighbor space-time cluster detection method. Written by Nicholas Malizia on 7/2/2010. 

"""

import pysal
from pysal.common import *

#Declares the input data. Shapefile must be composed of points. Time stamp is found in the DBF. 
path = '/Users/Nick/Documents/Academic/Data/SpatialDynamics/burkitt'
shp = pysal.open(path + '.shp')
dbf = pysal.open(path + '.dbf')
timecol = 'T'
k = 4


#Extract the spatial coordinates from the shapefile and combine into a numpy array.
x = []
y = []
n = 0
for i in shp:
    count = 0
    for j in i:
        if count==0:
            x.append(j)
        elif count==1:
            y.append(j)
        count += 1
    n += 1 

x = np.array(x)
x = np.reshape(x,(n,1))
y = np.array(y)
y = np.reshape(y,(n,1)) 
space = np.hstack((x,y))


#Extract the temporal information from the dbf and place it on a number line. 
t = np.array(dbf.by_col(timecol))
t = np.reshape(t,(n,1))
line = np.ones((n,1))
time = np.hstack((t,line))


#Calculate the nearest neighbors in space and time separately. 
nntime = pysal.knnW(time, k)
nnspace = pysal.knnW(space, k)
knn_ids = {}
knn_sum = 0
#There is a problem here associated with the KNN function. It breaks whenever there are coincident points (in space or time). This stems from the KDTree within the knnW function. Ask Serge about this. 


#Determine which events are nearest neighbors in both space and time. 
for i in range(n):
    t_neighbors = nntime.neighbors[i]
    s_neighbors = nnspace.neighbors[i]
    check = set(t_neighbors)
    inter = check.intersection(s_neighbors)
    count = len(inter)
    inter = list(inter)
    #Store the count and the ids for the matches. 
    knn_sum += count
    knn_ids[i] = inter





    


            

