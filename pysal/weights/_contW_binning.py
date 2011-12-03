#!/usr/bin/python
#import math
import pysal
from pysal.cg.standalone import get_shared_segments

__author__  = "Sergio J. Rey <srey@asu.edu> "
__all__ = ["QUEEN", "ROOK", "ContiguityWeights_binning"]


# delta to get buckets right
DELTA = 0.000001

QUEEN = 1
ROOK = 2

# constants for bucket sizes
BUCK_SM = 8
BUCK_LG = 80
SHP_SMALL = 1000

def bbcommon(bb, bbother):
    """
    Checks for overlaps of bounding boxes. First, east-west, then north-south. 
    Element 0 is west, element 2 is east, element 1 is north?, element 3 is
    south?
    All four checks must be false for chflag to be true, meaning the two
    bounding boxes do not overlap.
    """
    chflag = 0
    if not ((bbother[2] < bb[0]) or (bbother[0] > bb[2])):
        if not ((bbother[3] < bb[1]) or (bbother[1] > bb[3])):
            chflag = 1
    return chflag

class ContiguityWeights_binning:
    """ """
    def __init__(self, shpFileObject, wttype):
        self.shpFileObject = shpFileObject
        self.wttype = wttype
        self.bucket()
        self.doWeights()

    def bucket(self):
        shpFileObject = self.shpFileObject

        if shpFileObject.type != pysal.cg.Polygon:
            return False

        shapebox = shpFileObject.bbox      # bounding box

        numPoly = len(shpFileObject)
        self.numPoly = numPoly

        # bucket size
        if (numPoly < SHP_SMALL):
            bucketmin = numPoly / BUCK_SM + 2
        else:
            bucketmin = numPoly / BUCK_LG + 2
        # bucket length
        lengthx = ((shapebox[2]+DELTA) - shapebox[0]) / bucketmin
        lengthy = ((shapebox[3]+DELTA) - shapebox[1]) / bucketmin
        
        # initialize buckets
        columns = [ set() for i in range(bucketmin) ]
        rows = [ set() for i in range(bucketmin) ]
                
        minbox = shapebox[:2] * 2                                  # minx,miny,minx,miny
        binWidth = [lengthx, lengthy] * 2                              # lenx,leny,lenx,leny
        bbcache = {} 
        poly2Column = [ set() for i in range(numPoly) ]
        poly2Row = [ set() for i in range(numPoly) ]
        for i in range(numPoly):
            shpObj = shpFileObject.get(i)
            bbcache[i] = shpObj.bounding_box[:]
            projBBox = [int((shpObj.bounding_box[:][j] - minbox[j])/binWidth[j]) for j in xrange(4)]
            for j in range(projBBox[0], projBBox[2]+1):
                columns[j].add(i)
                poly2Column[i].add(j)
            for j in range(projBBox[1], projBBox[3]+1):
                rows[j].add(i)
                poly2Row[i].add(j)
        # loop over polygons rather than bins
        w = {}
        for polyId in xrange(numPoly):
            idRows = poly2Row[polyId]
            idCols = poly2Column[polyId]
            rowPotentialNeighbors = set()
            colPotentialNeighbors = set()
            for row in idRows:
                rowPotentialNeighbors = rowPotentialNeighbors.union(rows[row])
            for col in idCols:
                colPotentialNeighbors = colPotentialNeighbors.union(columns[col])
            potentialNeighbors = rowPotentialNeighbors.intersection(colPotentialNeighbors)
            if polyId not in w:
                w[polyId] = set()
            for j in potentialNeighbors:
                if polyId < j:
                    if bbcommon(bbcache[polyId], bbcache[j]):
                        w[polyId].add(j)

        self.potentialW = w

    def doWeights(self):
        pw = self.potentialW
        polygonCache = {}
        w = {}
        shpFileObject = self.shpFileObject
        for polyId in xrange(self.numPoly):
            if polyId not in polygonCache:
                iVerts = set(shpFileObject.get(polyId).vertices)
                polygonCache[polyId] = iVerts
            else:
                iVerts = polygonCache[polyId]
            potentialNeighbors = pw[polyId]
            if polyId not in w:
                w[polyId] = set()
            for j in potentialNeighbors:
                if j not in polygonCache:
                    polygonCache[j] = set(shpFileObject.get(j).vertices)
                common = iVerts.intersection(polygonCache[j])
                join = False
                if len(common) > 1: #ROOK
                    #double check rook
                    poly0 = shpFileObject.get(polyId)
                    poly1 = shpFileObject.get(j)
                    if get_shared_segments(poly0,poly1,True):
                        join = True
                    #for vert in common:
                    #    idx = poly0.vertices.index(vert)
                    #    IDX = poly1.vertices.index(vert)
                    #    try:
                    #        if poly0.vertices[idx+1] == poly1.vertices[IDX+1] or poly0.vertices[idx+1] == poly1.vertices[IDX-1]\
                    #        or poly0.vertices[idx-1] == poly1.vertices[IDX+1] or poly0.vertices[idx-1] == poly1.vertices[IDX-1]:
                    #            join = True
                    #            break
                    #    except IndexError:
                    #        pass
                if len(common) > 0: #QUEEN
                    if self.wttype == QUEEN:
                        join = True
                if join:
                    w[polyId].add(j)
                    if j not in w:
                        w[j] = set()
                    w[j].add(polyId)

            del polygonCache[polyId]
        self.w = w

if __name__ == "__main__":
    import time
    fname = pysal.examples.get_path('10740.shp')
    t0 = time.time()
    c = ContiguityWeights_binning(pysal.open(fname), QUEEN)
    t1 = time.time()
    print "using "+str(fname)
    print "time elapsed for ... using bins: " + str(t1-t0)

