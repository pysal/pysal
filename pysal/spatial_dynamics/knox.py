"""
Modified Knox methods (raw codes)
"""

import random
import math

xytSet = []
for index in range(10):
    newX = random.random()
    newY = random.random()
    newT = random.randint(0, 100)
    single = (newX, newY, newT)
    xytSet.append(single)
#print"xytSet",xytSet

totalPair = 0
sumDistance = 0
sumAbsTime = 0
for index1 in range(len(xytSet) - 1):
    for index2 in range(index1 + 1, len(xytSet)):
        sumDistance += math.sqrt((xytSet[index1][0] - xytSet[index2][0]) * (xytSet[index1][0] - xytSet[index2][0])
                                  + (xytSet[index1][1] - xytSet[index2][1]) * (xytSet[index1][1] - xytSet[index2][1]))
        sumAbsTime += math.fabs(xytSet[index1][2] - xytSet[index2][2])
        totalPair += 1

averageDistance = sumDistance/totalPair
averageTime = sumAbsTime/totalPair


permutationMaxTimes = 1
for index in range(len(xytSet) + 1):
    if index == 0:
        pass
    else:
        permutationMaxTimes *= index
if permutationMaxTimes > 999:
    permutationMaxTimes = 999

#extract time
timeSet = []
for index in range(len(xytSet)):
    timeSet.append(xytSet[index][2])
#print timeSet

#recursive function
def Getpermutation(single):
    if len(single) <=1:
        yield single
    else:
        for perm in Getpermutation(single[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + single[0:1] + perm[i:]


calPairs = []
count = 1
for oneTime in Getpermutation(timeSet):
    if count <= permutationMaxTimes:
        totalPair = 0
        for index1 in range(len(xytSet) - 1):
            for index2 in range(index1 + 1, len(xytSet)):
                distance = math.sqrt((xytSet[index1][0] - xytSet[index2][0]) * (xytSet[index1][0] - xytSet[index2][0])
                                       + (xytSet[index1][1] - xytSet[index2][1]) * (xytSet[index1][1] - xytSet[index2][1]))
                absTime = math.fabs(oneTime[index1] - oneTime[index2])
                if distance <= averageDistance and absTime <= averageTime:
                    totalPair += 1
        calPairs.append(totalPair)
        count += 1
    else:
        break

basicPair = calPairs[0]
calPairs = sorted(calPairs, reverse = True)
while index < len(calPairs):
    if basicPair >= calPairs[index]:
        break
    index += 1
sl=(index+1)*1./(permutationMaxTimes+1)
print "significance level",sl
