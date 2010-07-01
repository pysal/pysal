"""
LISA Time Path methods (raw codes)
"""

import random
import math

xySet = []
for index in range(10):
    newX = random.random()
    newY = random.random()
    coord = (newX, newY)
    xySet.append(coord)
#print xySet

#the distance between the first and the last point
absoluteDistance = math.sqrt((xySet[0][0] - xySet[-1][0]) * (xySet[0][0] - xySet[-1][0])
                             + (xySet[0][1] - xySet[-1][1]) * (xySet[0][1] - xySet[-1][1]))
# the length of the chain
sumDistance = 0
for index in range(len(xySet)- 1):
    sumDistance += math.sqrt((xySet[index][0] - xySet[index + 1][0]) * (xySet[index][0] - xySet[index + 1][0])
                             + (xySet[index][1] - xySet[index + 1][1]) * (xySet[index][1] - xySet[index + 1][1]))

#Curve
curveDegree = absoluteDistance*1./sumDistance

#Count the movement Direction of each path segment
quadrant1 = 0
quadrant2 = 0
quadrant3 = 0
quadrant4 = 0
for index in range(len(xySet) - 1):
    if xySet[index][0] <= xySet[index + 1][0] and xySet[index][1] < xySet[index + 1][1]:
        quadrant1 += 1
    elif xySet[index][0] > xySet[index + 1][0] and xySet[index][1] <= xySet[index + 1][1]:
        quadrant2 += 1
    elif xySet[index][0] >= xySet[index + 1][0] and xySet[index][1] > xySet[index + 1][1]:
        quadrant3 += 1
    else:
        quadrant4 += 1
##print "quad1: ", quadrant1
##print "quad2: ", quadrant2
##print "quad3: ", quadrant3
##print "quad4: ", quadrant4

