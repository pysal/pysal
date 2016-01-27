#!/usr/bin/python
#import math
import pysal
from pysal.cg.standalone import get_shared_segments

__author__ = "Sergio J. Rey <srey@asu.edu> "
__all__ = ["QUEEN", "ROOK", "ContiguityWeights_binning",
           "ContiguityWeightsPolygons"]


import time

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
    Element 0 is west, element 2 is east, element 3 is north, element 1 is
    south.
    All four checks must be false for chflag to be true, meaning the two
    bounding boxes do not overlap.
    """
    chflag = 0
    if not ((bbother[2] < bb[0]) or (bbother[0] > bb[2])):
        if not ((bbother[3] < bb[1]) or (bbother[1] > bb[3])):
            chflag = 1
    return chflag


class ContiguityWeights_binning:

    """
    Contiguity using a binning algorithm
    """

    def __init__(self, shpFileObject, wttype):
        self.shpFileObject = shpFileObject
        self.wttype = wttype
        self.do_weights()

    def do_weights(self):
        shpFileObject = self.shpFileObject

        if shpFileObject.type != pysal.cg.Polygon:
            return False

        shapebox = shpFileObject.bbox      # bounding box

        numPoly = len(shpFileObject)
        self.numPoly = numPoly

        # bucket size
        if (numPoly < SHP_SMALL):
            bucketmin = numPoly // BUCK_SM + 2
        else:
            bucketmin = numPoly // BUCK_LG + 2
            # print 'bucketmin: ', bucketmin
        # bucket length
        lengthx = ((shapebox[2] + DELTA) - shapebox[0]) / bucketmin
        lengthy = ((shapebox[3] + DELTA) - shapebox[1]) / bucketmin

        # print lengthx, lengthy

        # initialize buckets
        columns = [set() for i in range(bucketmin)]
        rows = [set() for i in range(bucketmin)]

        minbox = shapebox[:2] * 2             # minx,miny,minx,miny
        binWidth = [lengthx, lengthy] * 2      # lenx,leny,lenx,leny
        bbcache = {}
        poly2Column = [set() for i in range(numPoly)]
        poly2Row = [set() for i in range(numPoly)]
        for i in range(numPoly):
            shpObj = shpFileObject.get(i)
            bbcache[i] = shpObj.bounding_box[:]
            projBBox = [int((shpObj.bounding_box[:][j] -
                             minbox[j]) / binWidth[j]) for j in xrange(4)]
            for j in range(projBBox[0], projBBox[2] + 1):
                columns[j].add(i)
                poly2Column[i].add(j)
            for j in range(projBBox[1], projBBox[3] + 1):
                rows[j].add(i)
                poly2Row[i].add(j)

        w = {}
        if self.wttype == QUEEN:
            # loop over polygons rather than bins
            vertCache = {}
            for polyId in xrange(numPoly):
                if polyId not in vertCache:
                    vertCache[polyId] = set(shpFileObject.get(polyId).vertices)
                idRows = poly2Row[polyId]
                idCols = poly2Column[polyId]
                rowPotentialNeighbors = set()
                colPotentialNeighbors = set()
                for row in idRows:
                    rowPotentialNeighbors = rowPotentialNeighbors.union(
                        rows[row])
                for col in idCols:
                    colPotentialNeighbors = colPotentialNeighbors.union(
                        columns[col])
                potentialNeighbors = rowPotentialNeighbors.intersection(
                    colPotentialNeighbors)
                if polyId not in w:
                    w[polyId] = set()
                for j in potentialNeighbors:
                    if polyId < j:
                        if bbcommon(bbcache[polyId], bbcache[j]):
                            if j not in vertCache:
                                vertCache[j] = set(
                                    shpFileObject.get(j).vertices)
                            common = vertCache[
                                polyId].intersection(vertCache[j])
                            if len(common) > 0:
                                w[polyId].add(j)
                                if j not in w:
                                    w[j] = set()
                                w[j].add(polyId)
        elif self.wttype == ROOK:
            # check for a shared edge
            edgeCache = {}
            # loop over polygons rather than bins
            for polyId in xrange(numPoly):
                if polyId not in edgeCache:
                    iEdges = {}
                    iVerts = shpFileObject.get(polyId).vertices
                    nv = len(iVerts)
                    ne = nv - 1
                    for i in xrange(ne):
                        l = iVerts[i]
                        r = iVerts[i + 1]
                        iEdges[(l, r)] = []
                        iEdges[(r, l)] = []
                    edgeCache[polyId] = iEdges
                iEdgeSet = set(edgeCache[polyId].keys())
                idRows = poly2Row[polyId]
                idCols = poly2Column[polyId]
                rowPotentialNeighbors = set()
                colPotentialNeighbors = set()
                for row in idRows:
                    rowPotentialNeighbors = rowPotentialNeighbors.union(
                        rows[row])
                for col in idCols:
                    colPotentialNeighbors = colPotentialNeighbors.union(
                        columns[col])
                potentialNeighbors = rowPotentialNeighbors.intersection(
                    colPotentialNeighbors)
                if polyId not in w:
                    w[polyId] = set()
                for j in potentialNeighbors:
                    if polyId < j:
                        if bbcommon(bbcache[polyId], bbcache[j]):
                            if j not in edgeCache:
                                jVerts = shpFileObject.get(j).vertices
                                jEdges = {}
                                nv = len(jVerts)
                                ne = nv - 1
                                for e in xrange(ne):
                                    l = jVerts[e]
                                    r = jVerts[e + 1]
                                    jEdges[(l, r)] = []
                                    jEdges[(r, l)] = []
                                edgeCache[j] = jEdges
                            # for edge in edgeCache[j]:
                            if iEdgeSet.intersection(edgeCache[j].keys()):
                                w[polyId].add(j)
                                if j not in w:
                                    w[j] = set()
                                w[j].add(polyId)
                                # break
        else:
            print "Unsupported weight type."

        self.w = w

# Generalize to handle polygon collections - independent of origin file type


class ContiguityWeightsPolygons:

    """
    Contiguity for a collection of polygons using a binning algorithm
    """

    def __init__(self, collection, wttype=1):
        """

        Parameters
        ==========

        collection: PySAL PolygonCollection 

        wttype: int
                1: Queen
                2: Rook
        """

        self.collection = collection
        self.wttype = wttype
        self.do_weights()

    def do_weights(self):
        if self.collection.type != pysal.cg.Polygon:
            return False

        shapebox = self.collection.bbox      # bounding box

        numPoly = self.collection.n
        self.numPoly = numPoly

        # bucket size
        if (numPoly < SHP_SMALL):
            bucketmin = numPoly // BUCK_SM + 2
        else:
            bucketmin = numPoly // BUCK_LG + 2
            # print 'bucketmin: ', bucketmin
        # bucket length
        lengthx = ((shapebox[2] + DELTA) - shapebox[0]) / bucketmin
        lengthy = ((shapebox[3] + DELTA) - shapebox[1]) / bucketmin

        # print lengthx, lengthy

        # initialize buckets
        columns = [set() for i in range(bucketmin)]
        rows = [set() for i in range(bucketmin)]

        minbox = shapebox[:2] * 2             # minx,miny,minx,miny
        binWidth = [lengthx, lengthy] * 2      # lenx,leny,lenx,leny
        bbcache = {}
        poly2Column = [set() for i in range(numPoly)]
        poly2Row = [set() for i in range(numPoly)]
        for i in range(numPoly):
            shpObj = self.collection[i]
            bbcache[i] = shpObj.bbox[:]
            projBBox = [int((shpObj.bbox[:][j] -
                             minbox[j]) / binWidth[j]) for j in xrange(4)]
            for j in range(projBBox[0], projBBox[2] + 1):
                columns[j].add(i)
                poly2Column[i].add(j)
            for j in range(projBBox[1], projBBox[3] + 1):
                rows[j].add(i)
                poly2Row[i].add(j)

        w = {}
        if self.wttype == QUEEN:
            # loop over polygons rather than bins
            vertCache = {}
            for polyId in xrange(numPoly):
                if polyId not in vertCache:
                    vertCache[polyId] = set(self.collection[polyId].vertices)
                idRows = poly2Row[polyId]
                idCols = poly2Column[polyId]
                rowPotentialNeighbors = set()
                colPotentialNeighbors = set()
                for row in idRows:
                    rowPotentialNeighbors = rowPotentialNeighbors.union(
                        rows[row])
                for col in idCols:
                    colPotentialNeighbors = colPotentialNeighbors.union(
                        columns[col])
                potentialNeighbors = rowPotentialNeighbors.intersection(
                    colPotentialNeighbors)
                if polyId not in w:
                    w[polyId] = set()
                for j in potentialNeighbors:
                    if polyId < j:
                        if j not in vertCache:
                            vertCache[j] = set(self.collection[j].vertices)
                        if bbcommon(bbcache[polyId], bbcache[j]):
                            vertCache[j] = set(self.collection[j].vertices)
                            common = vertCache[
                                polyId].intersection(vertCache[j])
                            if len(common) > 0:
                                w[polyId].add(j)
                                if j not in w:
                                    w[j] = set()
                                w[j].add(polyId)
        elif self.wttype == ROOK:
            # check for a shared edge
            edgeCache = {}
            # loop over polygons rather than bins
            for polyId in xrange(numPoly):
                if polyId not in edgeCache:
                    iEdges = {}
                    iVerts = shpFileObject.get(polyId).vertices
                    nv = len(iVerts)
                    ne = nv - 1
                    for i in xrange(ne):
                        l = iVerts[i]
                        r = iVerts[i + 1]
                        iEdges[(l, r)] = []
                        iEdges[(r, l)] = []
                    edgeCache[polyId] = iEdges
                iEdgeSet = set(edgeCache[polyId].keys())
                idRows = poly2Row[polyId]
                idCols = poly2Column[polyId]
                rowPotentialNeighbors = set()
                colPotentialNeighbors = set()
                for row in idRows:
                    rowPotentialNeighbors = rowPotentialNeighbors.union(
                        rows[row])
                for col in idCols:
                    colPotentialNeighbors = colPotentialNeighbors.union(
                        columns[col])
                potentialNeighbors = rowPotentialNeighbors.intersection(
                    colPotentialNeighbors)
                if polyId not in w:
                    w[polyId] = set()
                for j in potentialNeighbors:
                    if polyId < j:
                        if bbcommon(bbcache[polyId], bbcache[j]):
                            if j not in edgeCache:
                                jVerts = shpFileObject.get(j).vertices
                                jEdges = {}
                                nv = len(jVerts)
                                ne = nv - 1
                                for e in xrange(ne):
                                    l = jVerts[e]
                                    r = jVerts[e + 1]
                                    jEdges[(l, r)] = []
                                    jEdges[(r, l)] = []
                                edgeCache[j] = jEdges
                            # for edge in edgeCache[j]:
                            if iEdgeSet.intersection(edgeCache[j].keys()):
                                w[polyId].add(j)
                                if j not in w:
                                    w[j] = set()
                                w[j].add(polyId)
                                # break
        else:
            print "Unsupported weight type."

        self.w = w

if __name__ == "__main__":
    import time
    fname = pysal.examples.get_path('NAT.shp')
    print 'QUEEN binning'
    t0 = time.time()
    qb = ContiguityWeights_binning(pysal.open(fname), QUEEN)
    t1 = time.time()
    print "using " + str(fname)
    print "time elapsed for queen... using bins: " + str(t1 - t0)

    t0 = time.time()
    rb = ContiguityWeights_binning(pysal.open(fname), ROOK)
    t1 = time.time()
    print 'Rook binning'
    print "using " + str(fname)
    print "time elapsed for rook... using bins: " + str(t1 - t0)

    from pysal.weights._contW_rtree import ContiguityWeights_rtree

    t0 = time.time()
    rt = ContiguityWeights_rtree(pysal.open(fname), ROOK)
    t1 = time.time()

    print "time elapsed for rook... using rtree: " + str(t1 - t0)
    print rt.w == rb.w

    print 'QUEEN'
    t0 = time.time()
    qt = ContiguityWeights_rtree(pysal.open(fname), QUEEN)
    t1 = time.time()
    print "using " + str(fname)
    print "time elapsed for queen... using rtree: " + str(t1 - t0)
    print qb.w == qt.w

    print 'knn4'
    t0 = time.time()
    knn = pysal.knnW_from_shapefile(fname, k=4)
    t1 = time.time()
    print t1 - t0

    print 'rook from shapefile'
    t0 = time.time()
    knn = pysal.rook_from_shapefile(fname)
    t1 = time.time()
    print t1 - t0
