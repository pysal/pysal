#!/usr/bin/python

import time
import os
import math
import shpIO
from shapeReader import *

# delta to get buckets right
DELTA = 0.000001

# constants for weights types
WT_UNKNOWN = 0
WT_ROOK = 1
WT_QUEEN = 2

# constants for bucket sizes
BUCK_SM = 8
BUCK_LG = 80
SHP_SMALL = 1000

def bbcommon(bb,bbother):
    chflag = 0
    if not ((bbother[2] < bb[0]) or (bbother[0] > bb[2])):
        if not ((bbother[3] < bb[1]) or (bbother[1] > bb[3])):
            chflag = 1
    return chflag


class spweights:
    """
    spweights data structure: list of lists
    [ metadata ], [key:obs dictionary] [obs:key dictionary] [neighbor id lists ] [neighbor weights lists]
    [characteristics ] [traces ] [eigvalues]
    need a dbf reader to use a "key" variable instead of sequence number
    sequence number is fine as long as "pure" shape files since the matching order
    is part of the ESRI shape file format
    """
    
    
    def __init__(self,filename, outfilename, wtType):
    # initialization here: check input file type and
    # call appropriate reader
        self.meta = []
        self.keyobs = {}
        self.obskey = {}
        self.neighbors = []
        self.weights  = []
        self.characteristics = []
        self.traces = []
        self.eigvalues = []
        # only shape input implemented so far
        self.shp2wt(filename,wtType)
        #self.weight2file(filename, outfilename, wtType)

    # save weight as a file
    def weight2file(self, filename, outname, wtType):
        shppath = os.path.split(filename)
        shptitle = shppath[1]
        
        output = open(outname,'w')

        if (wtType == 1):
            typeTitle = "Rook Contiguity"
        elif (wtType == 2):
            typeTitle = "Queen Contiguity"
        elif (wtType == 3):
            typeTitle = "Threshold Distance"
        elif (wtType == 4):
            typeTitle = "K-Nearest Neighbors"
            
        output.writelines(str(len(self.neighbors))+ " " + shptitle + " " + typeTitle + "\n")
        for i in range(len(self.neighbors)):
            output.writelines(str(i+1) + " " + str(len(self.neighbors[i]))+ "\n")
            for j in range(len(self.neighbors[i])):
                output.writelines(str(self.neighbors[i][j] + 1) + " ")
            output.writelines("\n")
        output.close()
        
    # read gal file and convert to wt data structure
    def wtfromgal(self):
        pass
        
    # read gwt file and convert to wt data structure
    def wtfromgwt(self):
        pass
        
    # read shp file and construct rook or queen contiguity
    def shp2wt(self,filename,wtType):
        raw_shape = shapefile(filename)
        if raw_shape.shptype == SHP_POINT:
            return       # need handler 
        shapepoints = raw_shape.shplist    # list of lists
        shapebox = raw_shape.shpbox      # bounding box
        
        numPoly = len(shapepoints)
        # bucket size
        if (numPoly < SHP_SMALL):
            bucketMin = numPoly / BUCK_SM + 2
        else:
            bucketMin = numPoly / BUCK_LG + 2
        # bucket length
        lengthX = ((shapebox[2]+DELTA) - shapebox[0]) / bucketMin
        lengthY = ((shapebox[3]+DELTA) - shapebox[1]) / bucketMin
        #print "lengthX: ", lengthX
        #print "bucketMin: ",bucketMin
        
        # initialize buckets
        bucketX = [ [] for i in range(bucketMin) ]
        bucketY = [ [] for i in range(bucketMin) ]
        polyXbucket = [ [] for i in range(numPoly) ]              # list with buckets for X
        polyYbucket = [ [] for i in range(numPoly) ]              # list with buckets for Y
        self.neighbors = [ [] for i in range(numPoly) ]         # list of lists for neighbors
                
        minbox = shapebox[:2] * 2                                  # minX,minY,minX,minY
        blen = [lengthX,lengthY] * 2                              # lenx,leny,lenx,leny
        
        for i in range(numPoly):
            pb = [int((shapepoints[i][0][j] - minbox[j])/blen[j]) for j in range(4)]
            for j in range(pb[0],pb[2]+1):
                polyXbucket[i].append(j)
                bucketX[j].append(i)
            for j in range(pb[1],pb[3]+1):
                polyYbucket[i].append(j)
                bucketY[j].append(i)
            
        #create candidate neighbors from buckets
        if False:
        #for i in range(numPoly):
            buckX = []
            for j in range(0,len(polyXbucket[i])):
                buckX += bucketX[polyXbucket[i][j]]
            buckY = []
            for j in range(0,len(polyYbucket[i])):
                buckY += bucketY[polyYbucket[i][j]]
            buckX = dict( [ (j,j) for j in buckX ]).keys()
            buckY = dict( [ (j,j) for j in buckY ]).keys()
            buckX.sort()
            buckY.sort()
            nb = []
            if len(buckX) < len(buckY):
                k = buckX.index(i) + 1
                nb = [ buckX[jj] for jj in range(k,len(buckX)) 
                        if (buckX[jj] in buckY) and (shapepoints[i][0].bbcommon(shapepoints[buckX[jj]][0]) ) ]
            else:
                k = buckY.index(i) + 1
                nb = [ buckY[jj] for jj in range(k,len(buckY)) if  (buckY[jj] in buckX)
                        and (shapepoints[i][0].bbcommon(shapepoints[buckY[jj]][0])) ]
            for ii in range(0,len(nb)):
                ch=0
                jj=0
                kk=len(shapepoints[i][2]) - 1
                nbi = nb[ii]
                while not ch and jj < kk:
                    if wtType == WT_ROOK:
                        ch = (shapepoints[i][2][jj] in shapepoints[nbi][2]) and (shapepoints[i][2][jj+1] in shapepoints[nbi][2])
                    else:   # queen
                        ch = shapepoints[i][2][jj] in shapepoints[nbi][2]
                    jj += 1
                if ch:
                    self.neighbors[i].append(nbi)
                    self.neighbors[nbi].append(i)



                    

########################################
    # read pickled weight file ?
    def readwt(self):
        pass
        
    # write wt to gal file
    def wt2gal(self):
        pass
        
    # write wt to gwt file
    def wt2gwt(self):
        pass
        
    # pickle wt file
    def wt2pickle(self):
        pass
        
    # write numpy to gal file
    def mat2gal(self):
        pass
        
    # write numpty to gwt file
    def mat2gwt(self):
        pass
        
    # weights characteristics
    def wtchars(self):
        pass
        
    # weights traces
    def wttraces(self):
        pass
        
    # higher order weights
    def wt2higher(self):
        pass
        
    # conversion from wt data structure to numpy weight matrix
    def wt2mat(self):
        pass
        
    # weights eigenvalues
    def wteigen(self):
        pass
    
    # distance weights
    
    # spatial lag
    
    # spatial filter
    
    # spatial AR transformation
    
    # editing weights
    
    # visualizing the  structure of weights
    
    # weights computations: addition, subtraction, multiplication

class spweightsl:
    """
    spweights data structure: list of lists
    [ metadata ], [key:obs dictionary] [obs:key dictionary] [neighbor id lists ] [neighbor weights lists]
    [characteristics ] [traces ] [eigvalues]
    need a dbf reader to use a "key" variable instead of sequence number
    sequence number is fine as long as "pure" shape files since the matching order
    is part of the ESRI shape file format
    """
    
    
    def __init__(self,filename, outfilename, wtType):
    # initialization here: check input file type and
    # call appropriate reader
        self.meta = []
        self.keyobs = {}
        self.obskey = {}
        self.neighbors = []
        self.weights  = []
        self.characteristics = []
        self.traces = []
        self.eigvalues = []
        # only shape input implemented so far
        self.shp2wt(filename,wtType)
        #self.weight2file(filename, outfilename, wtType)

    # save weight as a file
    def weight2file(self, filename, outname, wtType):
        shppath = os.path.split(filename)
        shptitle = shppath[1]
        
        output = open(outname,'w')

        if (wtType == 1):
            typeTitle = "Rook Contiguity"
        elif (wtType == 2):
            typeTitle = "Queen Contiguity"
        elif (wtType == 3):
            typeTitle = "Threshold Distance"
        elif (wtType == 4):
            typeTitle = "K-Nearest Neighbors"
            
        output.writelines(str(len(self.neighbors))+ " " + shptitle + " " + typeTitle + "\n")
        for i in range(len(self.neighbors)):
            output.writelines(str(i+1) + " " + str(len(self.neighbors[i]))+ "\n")
            for j in range(len(self.neighbors[i])):
                output.writelines(str(self.neighbors[i][j] + 1) + " ")
            output.writelines("\n")
        output.close()
        
    # read gal file and convert to wt data structure
    def wtfromgal(self):
        pass
        
    # read gwt file and convert to wt data structure
    def wtfromgwt(self):
        pass
        
    # read shp file and construct rook or queen contiguity
    def shp2wt(self,filename,wtType):
        raw_shape = shapefile(filename,large=1)
        if raw_shape.shptype == SHP_POINT:
            return       # need handler 
        shapepoints = raw_shape.shplist    # list of lists
        shapebox = raw_shape.shpbox      # bounding box
        
        pointDict={}

        for i,shape in enumerate(shapepoints):
            #print i
            for point in shape[2]:
                if not point in pointDict:
                    pointDict[point]=set()
                pointDict[point].add(i)

        w={}
        for point in pointDict:
            nn=len(pointDict[point])
            if nn > 1:
                neighbors=pointdict[point]
                for i in neighbors:
                    w[i]=w.get(i,{})
                    for j in neighbors:
                        if i!=j:
                            w[i][j]=w[i].get(j,0)+1


        """
                for i in xrange(nn-1):
                    for j in xrange(i+1,nn):
                        try:
                            w[i].append(j)
                            w[(i,j)]=1
                            w[(j,i)]=1
                        except:
                            w[(i,j)]=+1
                            w[(j,i)]=+1
                while neighbors:
                    i=neighbors.pop()
                    ijs=[ (i,j) for j in neighbors]
                    while ijs:
                        i,j = ijs.pop()
                        try:
                            w[(i,j)]+=1
                            w[(j,i)]+=1
                        except:
                            w[(i,j)]=1
                            w[(j,i)]=1
        """
        if wtType==1:
            wr={}
            for i in w:
                wr[i]=[ j for j in w[i] if w[i][j]>1]
            self.w=wr
        elif wtType==2:
            self.w=w


                    

########################################
    # read pickled weight file ?
    def readwt(self):
        pass
        
    # write wt to gal file
    def wt2gal(self):
        pass
        
    # write wt to gwt file
    def wt2gwt(self):
        pass
        
    # pickle wt file
    def wt2pickle(self):
        pass
        
    # write numpy to gal file
    def mat2gal(self):
        pass
        
    # write numpty to gwt file
    def mat2gwt(self):
        pass
        
    # weights characteristics
    def wtchars(self):
        pass
        
    # weights traces
    def wttraces(self):
        pass
        
    # higher order weights
    def wt2higher(self):
        pass
        
    # conversion from wt data structure to numpy weight matrix
    def wt2mat(self):
        pass
        
    # weights eigenvalues
    def wteigen(self):
        pass
    
    # distance weights
    
    # spatial lag
    
    # spatial filter
    
    # spatial ar transformation
    
    # editing weights
    
    # visualizing the  structure of weights
    
    # weights computations: addition, subtraction, multiplication

class ContiguityWeights:
    """ """
    def __init__(self, shpFileObject, outfileobject, wttype):
        self.shpFileObject=shpFileObject
        self.bucket()
        self.doWeights()
        self.fileName=shpFileObject.fileName

    def bucket(self):
        shpFileObject=self.shpFileObject

        if shpFileObject.shapeType!=shpIO.POLYGON:
            return false

        shapebox = shpFileObject.bbox      # bounding box


        numPoly = shpFileObject.numRecords
        self.numPoly=numPoly

        # bucket size
        if (numPoly < SHP_SMALL):
            bucketmin = numPoly / BUCK_SM + 2
        else:
            bucketmin = numPoly / BUCK_LG + 2
        # bucket length
        lengthx = ((shapebox[2]+DELTA) - shapebox[0]) / bucketmin
        lengthy = ((shapebox[3]+DELTA) - shapebox[1]) / bucketmin
        #print "lengthx: ", lengthx
        #print "bucketmin: ",bucketmin
        
        # initialize buckets
        columns = [ set() for i in range(bucketmin) ]
        rows = [ set() for i in range(bucketmin) ]
                
        minbox = shapebox[:2] * 2                                  # minx,miny,minx,miny
        binWidth = [lengthx,lengthy] * 2                              # lenx,leny,lenx,leny
        bbcache={} 
        poly2Column=[ set() for i in range(numPoly) ]
        poly2Row=[ set() for i in range(numPoly) ]
        for i in range(numPoly):
            shpObj=shpFileObject.get(i)
            bbcache[i]=shpObj.bbox
            projBBox=[int((shpObj.bbox[j] - minbox[j])/binWidth[j]) for j in xrange(4)]
            for j in range(projBBox[0],projBBox[2]+1):
                columns[j].add(i)
                poly2Column[i].add(j)
            for j in range(projBBox[1],projBBox[3]+1):
                rows[j].add(i)
                poly2Row[i].add(j)
        # loop over polygons rather than bins
        w={}
        for polyId in xrange(numPoly):
            idRows=poly2Row[polyId]
            idCols=poly2Column[polyId]
            rowPotentialNeighbors=set()
            colPotentialNeighbors=set()
            for row in idRows:
                rowPotentialNeighbors=rowPotentialNeighbors.union(rows[row])
            for col in idCols:
                colPotentialNeighbors=colPotentialNeighbors.union(columns[col])
            potentialNeighbors=rowPotentialNeighbors.intersection(colPotentialNeighbors)
            if polyId not in w:
                w[polyId]=set()
            for j in potentialNeighbors:
                if polyId < j:
                    if bbcommon(bbcache[polyId],bbcache[j]):
                        w[polyId].add(j)

        self.potentialW=w

    def doWeights(self):
        pw=self.potentialW
        polygonCache={}
        w={}
        shpFileObject=self.shpFileObject
        for polyId in xrange(self.numPoly):
            #print "polyId: ",polyId, "cache size: ",len(polygonCache)
            if polyId not in polygonCache:
                iVerts=shpFileObject.get(polyId).getVerticeSet()
                polygonCache[polyId]=iVerts
            else:
                iVerts=polygonCache[polyId]
            potentialNeighbors=pw[polyId]
            if polyId not in w:
                w[polyId]=set()
            for j in potentialNeighbors:
                if j not in polygonCache:
                    polygonCache[j]=shpFileObject.get(j).getVerticeSet()
                if iVerts.intersection(polygonCache[j]):
                    w[polyId].add(j)
                    if j not in w:
                        w[j]=set()
                    w[j].add(polyId)

            #print polygonCache.keys()
            del polygonCache[polyId]
            #print polygonCache.keys()
            #t=raw_input('here')
        self.w=w

    def writeGAL(self, fileName, newStyle=True):
        f=open(fileName,'w')
        if newStyle:
            header="0 %d %s %s\n"%(self.numPoly,self.fileName,'ID')
        else:
            header="%d\n"%self.numPoly
        f.write(header)

        # need to handle different types for the ids
        for i,n in self.w.items():
            nn=len(n)
            nlist=" ".join(["%s"% ni for ni in n])
            f.write("%d %d\n%s\n"%(i,nn,nlist))
        f.close()


        

#-----------------------------------------------------------------
class ptweights:
    """
    create point based weights, such as threshold distance, k-nearest neighbor
    accept raw shape from SAL_ShpReader, centroid list, and number of k
    ouput weight file will be generated
    """
    
    def __init__(self, raw_shape, shpFileName, weightFileName, weightFileClass, shpCentroid, numVar):
        # verify boudning box
        tmpXcoord = []
        tmpYcoord = []

        for i in range(len(shpCentroid)):
            tmpXcoord.append(shpCentroid[i][0])
            tmpYcoord.append(shpCentroid[i][1])

        shpBox = []
        shpBox.append(min(tmpXcoord))
        shpBox.append(min(tmpYcoord))
        shpBox.append(max(tmpXcoord))
        shpBox.append(max(tmpYcoord))

        tmpXcoord = []
        tmpYcoord = []
        
        # calculation
        if (weightFileClass == 3):            # threshold distance
            outResult = self.pt2threshold(shpBox, shpCentroid, numVar)
        elif (weightFileClass == 4):        # k-nerest neighbor
            outResult = self.pt2knearest(shpBox, shpCentroid, numVar)
            
        # save weight as a file
        self.weight2file(shpFileName, weightFileName, weightFileClass, outResult, numVar)

    def weight2file(self, filename, outname, wtType, outResult, numVar):
        """ create output weight file """
        shppath = os.path.split(filename)
        shptitle = shppath[1]
        
        output = open(outname,'w')

        if (wtType == 1):
            typeTitle = "Rook Contiguity"
        elif (wtType == 2):
            typeTitle = "Queen Contiguity"
        elif (wtType == 3):
            typeTitle = "Threshold Distance: " + str(numVar)
            output.writelines(str(len(outResult))+ " " + shptitle + " " + typeTitle + "\n")
            for i in range(len(outResult)):
                #output.writelines(str(i+1) + " " + str(len(outResult[i]))+ "\n")
                for j in range(len(outResult[i])):
                    output.writelines(str(i + 1) + " " + str(outResult[i][j][1] + 1) + "\t\t" + str(outResult[i][j][0]) + "\n")
            output.close()    

        # create output file for k-nearest neighbor            
        elif (wtType == 4):
            typeTitle = "K-Nearest Neighbors: " + str(numVar)
            output.writelines(str(len(outResult))+ " " + shptitle + " " + typeTitle + "\n")
            for i in range(len(outResult)):
                #output.writelines(str(i+1) + " " + str(numVar)+ "\n")
                for j in range(numVar):
                    output.writelines(str(i + 1) + " " + str(outResult[i][j][1] + 1) + "\t\t" + str(outResult[i][j][0]) + "\n")
            output.close()            

    def pt2threshold(self, shpBox, shpCentroid, tDistance):
        """ read coordinate list and construct threshold distance weight matrix """
        numPoint = len(shpCentroid)

        # origin point
        originX = shpBox[0]
        originY = shpBox[1]
        
        # bucket size should be square because the distance is the most important factor
        if (numPoint < SHP_SMALL):
            bucketMin = numPoint / BUCK_SM + 2
        else:
            bucketMin = numPoint / BUCK_LG + 2

        # calculate horizontal, vertical length of the shapefile
        lengthX = (shpBox[2] - shpBox[0])
        lengthY = (shpBox[3] - shpBox[1])

        if (lengthX > lengthY):        # horizontal length is longer than vertical length
            bucketX = bucketMin
            lengthX = ((shpBox[2]+DELTA) - shpBox[0]) / bucketMin
            lengthY = lengthX
            bucketY = int(((shpBox[3]+DELTA) - shpBox[1]) / lengthY) + 1
            
        else:
            bucketY = bucketMin
            lengthY = ((shpBox[3]+DELTA) - shpBox[1]) / bucketMin
            lengthX = lengthY
            bucketX = int(((shpBox[2]+DELTA) - shpBox[0]) / lengthX) + 1

        # initialize bucket
        bMatrix = [ [[] for i in range(bucketX)] for i in range(bucketY)]
        
        # distribute points to the bucket
        for i in range(numPoint):
            # figure out x bucket location
            tmpX = shpCentroid[i][0] - originX
            tmpX = int(tmpX / lengthX)
            
            # figure out y bucket location
            tmpY = shpCentroid[i][1] - originY
            tmpY = int(tmpY / lengthY)
        
            bMatrix[tmpY][tmpX].append(i)

        # construct the weight list
        outResult = []
        for i in range(numPoint):
            # figure out x bucket location
            tmpX = shpCentroid[i][0] - originX
            tmpX = int(tmpX / lengthX)
            
            # figure out y bucket location
            tmpY = shpCentroid[i][1] - originY
            tmpY = int(tmpY / lengthY)

            # verifying the number of buckets to go            
            tmpDepth = 1
            while ((tmpDepth * lengthX) <= tDistance):
                tmpDepth = tmpDepth + 1        

            # create temporary list with one level deeper for prevention edge error            
            tmpBin = self.createTmpBin(tmpDepth, bucketX, bucketY, bMatrix, tmpX, tmpY)
            
            # calculate the distance
            tmpResult = self.calcThresholdDistance(i, tDistance, tmpBin, shpCentroid)
            outResult.append(tmpResult)

        return outResult

    def calcThresholdDistance(self, iNum, tDistance, tmpBin, shpCentroid):
        """ calculate the euclidean distance between points for threshold distance"""
        tmpResult = []
        
        for i in range(len(tmpBin)):
            if (iNum != tmpBin[i]):
                lengthX = shpCentroid[iNum][0] - shpCentroid[tmpBin[i]][0]
                lengthY = shpCentroid[iNum][1] - shpCentroid[tmpBin[i]][1]
                distXY = math.sqrt(math.pow(lengthX, 2) + math.pow(lengthY, 2))
                if (distXY <= tDistance):
                    tmpResult.append((distXY, tmpBin[i]))
            
        return tmpResult    
                
    def pt2knearest(self, shpBox, shpCentroid, kNum):
        """ read coordinate list and construct k-nearest neighbor weight matrix """
        numPoint = len(shpCentroid)

        # origin point
        originX = shpBox[0]
        originY = shpBox[1]
        
        # bucket size should be square because the distance is the most important factor
        if (numPoint < SHP_SMALL):
            bucketMin = numPoint / BUCK_SM + 2
        else:
            bucketMin = numPoint / BUCK_LG + 2

        # calculate horizontal, vertical length of the shapefile
        lengthX = (shpBox[2] - shpBox[0])
        lengthY = (shpBox[3] - shpBox[1])

        if (lengthX > lengthY):        # horizontal length is longer than vertical length
            bucketX = bucketMin
            lengthX = ((shpBox[2]+DELTA) - shpBox[0]) / bucketMin
            lengthY = lengthX
            bucketY = int(((shpBox[3]+DELTA) - shpBox[1]) / lengthY) + 1
            
        else:
            bucketY = bucketMin
            lengthY = ((shpBox[3]+DELTA) - shpBox[1]) / bucketMin
            lengthX = lengthY
            bucketX = int(((shpBox[2]+DELTA) - shpBox[0]) / lengthX) + 1

        # initialize bucket
        bMatrix = [ [[] for i in range(bucketX)] for i in range(bucketY)]
        
        # distribute points to the bucket
        for i in range(numPoint):
            # figure out x bucket location
            tmpX = shpCentroid[i][0] - originX
            tmpX = int(tmpX / lengthX)
            
            # figure out y bucket location
            tmpY = shpCentroid[i][1] - originY
            tmpY = int(tmpY / lengthY)

            bMatrix[tmpY][tmpX].append(i)
        
        # construct the weight list
        outResult = []
        for i in range(numPoint):
            # figure out x bucket location
            tmpX = shpCentroid[i][0] - originX
            tmpX = int(tmpX / lengthX)
            
            # figure out y bucket location
            tmpY = shpCentroid[i][1] - originY
            tmpY = int(tmpY / lengthY)

            # verifying the depth of the difusion            
            tmpDepth = self.verifyKnearestDepth(kNum, bucketX, bucketY, bMatrix, tmpX, tmpY)

            # create temporary list with one level deeper for prevention edge error            
            tmpBin = self.createTmpBin(tmpDepth + 1, bucketX, bucketY, bMatrix, tmpX, tmpY)

            # calculate the distance
            tmpResult = self.calcKnearestDistance(i, tmpBin, shpCentroid)
            outResult.append(tmpResult)

        return outResult

    def verifyKnearestDepth(self, kNum, bucketX, bucketY, bMatrix, tmpX, tmpY):
        """ veryfiy how many depth of the bucket should be dig """
        j = 0
        tmpBin = bMatrix[tmpY][tmpX]

        while (len(tmpBin) < kNum + 1):
            tmpBin = []
            
            startX = tmpX - j
            if (startX < 0):
                startX = 0
            startY = tmpY - j
            if (startY < 0):
                startY = 0
            endX = tmpX + j + 1
            if (endX > bucketX):
                endX = bucketX
            endY = tmpY + j + 1
            if (endY > bucketY):
                endY = bucketY
        
            for k in range(startY, endY):
                for l in range(startX, endX):
                    tmpBin = tmpBin + bMatrix[k][l]

            j = j + 1

        return j    

    def calcKnearestDistance(self, iNum, tmpBin, shpCentroid):
        """ calculate the euclidean distance between points for k-nearest neighbor"""
        tmpResult = []
        
        for i in range(len(tmpBin)):
            if (iNum != tmpBin[i]):
                lengthX = shpCentroid[iNum][0] - shpCentroid[tmpBin[i]][0]
                lengthY = shpCentroid[iNum][1] - shpCentroid[tmpBin[i]][1]
                distXY = math.sqrt(math.pow(lengthX, 2) + math.pow(lengthY, 2))
                tmpResult.append((distXY, tmpBin[i]))
                tmpResult.sort()
            
        return tmpResult

    def createTmpBin(self, tmpDepth, bucketX, bucketY, bMatrix, tmpX, tmpY):
        """ construct tmporary bin that contains all the potential points from the bucket """
        tmpBin = []
        
        startX = tmpX - tmpDepth
        if (startX < 0):
            startX = 0
        startY = tmpY - tmpDepth
        if (startY < 0):
            startY = 0
        endX = tmpX + tmpDepth + 1
        if (endX > bucketX):
            endX = bucketX
        endY = tmpY + tmpDepth + 1
        if (endY > bucketY):
            endY = bucketY
    
        for k in range(startY, endY):
            for l in range(startX, endX):
                tmpBin = tmpBin + bMatrix[k][l]        
        
        return tmpBin    
    
    
#-----------------------------------------------------------------
class ReadGalFile:
    """"
    This is a class for reading contiguity weight file
    This should be revised and upgraded later
    """

    def __init__(self, filename):
        """ init object """
        self.neighbid = []
        self.neighbno = []
        self.readFile(filename)

    def readFile(self, galFile):
        """ read weight file and create object"""
        r_file = open(galFile, 'r')
        lines = r_file.read().split('\n')
        r_file.close()
        i_list = []
        n_list = []

        firstIdRow = 2
        firstNoRow = 1
        numRec = (len(lines) - 2) / 2

        for i in range(numRec):
            cells = lines[firstNoRow].split()
            n_list.append(int(cells[1]))
            firstNoRow = firstNoRow + 2
            cells = lines[firstIdRow].split()
            i_list.append(cells)
            firstIdRow = firstIdRow + 2

        del lines          
        del cells            
        self.neighbid = i_list
        self.neighbno = n_list

#------------------------------------------------------------------

class ReadGwtFile:
    """"
    This is a test class for reading distance weight
    This should be revised and upgraded later
    """

    def __init__(self, filename):
        """ init object """
        self.neighbid = []
        self.neighbno = []
        self.readFile(filename)

    def readFile(self, gwtFile):
        """ read weight file and create object """
        r_file = open(gwtFile, 'r')
        lines = r_file.read().split('\n')
        r_file.close()
        i_list = []
        n_list = []

        cells = lines[0].split()
        numRec = int(cells[0])
        numLine = (len(lines) - 2)
        
        for i in range(numRec):
            i_list.append([])
            n_list.append(0)

        for i in range(1, numLine + 1):
            cells = lines[i].split()
            n_list[int(cells[0]) - 1] = n_list[int(cells[0]) - 1] + 1
            i_list[int(cells[0]) - 1].append(cells[1])

        del lines          
        del cells            
        self.neighbid = i_list
        self.neighbno = n_list

#------------------------------------------------------------------    
# alternatively use subclasses with special constructors


if __name__ == "__main__":
    #fname = raw_input("Enter the shape file name (include .shp): ")
    fname="examples/usCounties/usa.shp"
    fout="examples/usCounties/test.out"
    t0 = time.time()
    w1=spweights(fname,fout,1)
    t1 = time.time()
    print "-------------------------------------"
    print "using "+str(fname)
    print "time elapsed for rook using bins: " + str(t1-t0)
    t2 = time.time()
    w2 = spweights(fname,fout,2)
    t3 = time.time()
    print "time elapsed for queen using bins: " + str(t3-t2)
    t1=time.time()
    w3=spweightsl(fname,fout,2)
    t2=time.time()
    print "time elapsed for queen using dicts: " + str(t2-t1)
    t1=time.time()
    w4=spweightsl(fname,fout,1)
    t2=time.time()
    print "time elapsed for rook using dicts: " + str(t2-t1)
    """

    fname="/Users/serge/Desktop/nhgis/us_tract_2000/US_tract_2000.shp"
    fout="/Users/serge/Desktop/nhgis/us_tract_2000/test.out"
    print "reading tracts" 
    t1=time.time()
    w4=spweightsl(fname,fout,2)
    t2=time.time()
    print "time elapsed for queen using dicts: " + str(t2-t1)
    fname="/Volumes/GeoDa/Projects/EDA/data/nhgis/us_tract_2000/US_tract_2000.shp"
    shpFile=shpIO.shpFile(fname)

    def run():
        c=ContiguityWeights(shpFile, None, None)
        
    import cProfile
    t0=time.time()
    cProfile.run('run()')
    t1=time.time()
    print t1-t0

    #c=ContiguityWeights(shpFile, None, None)
    #c.writeGAL("tracts2000.gal")
    """




    
