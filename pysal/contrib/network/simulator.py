#!/usr/bin/env python

"""
Author: Ran Wei, Myunghwa Hwang
"""

import pysal
import numpy as np

class Simulation(object):
   
    def __init__(self, src_filename):
        "create a bidirectional network with its total length and total number of links"
        self.nw = pysal.open(src_filename, 'r')
        self.G = {} # {edge_index:(d12,(n1,n2))}
        self.GNet = {} # {n1:{n2:d12}}
        self.total_length = 0.0
        self.nwNum = 0
        for line in self.nw:
            vertices = line.vertices
            for i, vertex in enumerate(vertices[:-1]):
                n1, n2 = vertex, vertices[i+1]
                self.G.setdefault(self.nwNum, ())
                self.GNet.setdefault(n1, {})
                self.GNet.setdefault(n2, {})
                d = pysal.cg.get_points_dist(pysal.cg.Point(n1), pysal.cg.Point(n2))
                self.G[self.nwNum] = (d, (n1, n2))
                self.GNet[n1][n2] = self.nwNum
                self.GNet[n2][n1] = self.nwNum
                self.total_length += d
                self.nwNum += 1
        self.nw.close()
        self.imaginaryLineGenerated = False
            
    def generateImaginaryLine(self):
        '''
           Create an imaginary line that starts from 0 and ends at 1
           and mark the locations of end points of each link
        '''
        self.nwCumPro = [0.0]
        for e in self.G.keys():
            self.nwCumPro.append(self.nwCumPro[-1] + (self.G[e][0]/self.total_length))
                                                                        
        # self.nwCumProDict --> {edge_index:right_side_end_point_of_the_link_on_the_imaginary_line}
        self.nwCumProDict = dict(zip(self.G.keys(), self.nwCumPro[1:])) 
        self.imaginaryLineGenerated = True
            
    def getRandomPoints(self, n, projected=False, toShp=False):   
        '''Create a random point pattern data set on the given network'''

        if not self.imaginaryLineGenerated:
            self.generateImaginaryLine()

        ## generate n unique random numbers between 0 and 1
        #randSet = set()
        #while len(randSet) < n: 
        #    randSet = set(np.random.random_sample(n))
        #randSet = np.array(list(randSet))        

        # generate n random numbers between 0 and 1
        randSet = np.random.random_sample(n)

        # Assign the random numbers to the links on the network
        # Think nwCumPro as bins; get bin numbers for all random numbers
        randSet_to_bins=np.digitize(randSet,self.nwCumPro)
        randSet_to_bins=zip(randSet_to_bins,randSet)
        randSet_to_bins.sort()
        # Determine geographic coordinates for each random number
        nwPtDict = {}
        for bin_id, rand_number in randSet_to_bins:
            bid = bin_id - 1
            n1, n2 = self.G[bid][1] # n1 and n2 are geographic (real) coordinates for the end points of a link
            origin = 0 if bid <= 0 else self.nwCumProDict[bid-1]
            length = self.nwCumProDict[bid] - origin
            # get prop to determine the geographic coordinate of a random number on the link (n1, n2)
            # length is the length of the link (n1, n2) on the imaginary line
            # (self.nwCumProDict[bin_id] - rand_number) is the distance between a random point and n2 
            # on the imaginary line
            prop = (self.nwCumProDict[bid] - rand_number)*1.0/length
            nwPtDict.setdefault(bid, [])
            if not projected:
                x = n2[0] - (n2[0] - n1[0])*prop # n2[0]: the geographic coordinate of n2 on the X axis
                y = n2[1] - (n2[1] - n1[1])*prop # n2[1]: the geographic coordinate of n2 on the Y axis
                nwPtDict[bid].append((x,y))
            else:
                dist = self.G[bid][0]
                proj_pnt = (n1, n2, dist*(1-prop), dist*prop)
                if toShp:
                    x = n2[0] - (n2[0] - n1[0])*prop
                    y = n2[1] - (n2[1] - n1[1])*prop
                    proj_pnt = tuple(list(proj_pnt) + [x, y])                        
                nwPtDict[bid].append(proj_pnt)
            
        return nwPtDict

    def countPointsOnNetwork(self, points, defaultBase=True):
        G = {}
        for k in self.G:
            n1, n2 = self.G[k][-1]
            G.setdefault(n1, {})
            G.setdefault(n2, {})
            if n2 not in G[n1]:
                attr = G[n1].setdefault(n2, [self.G[n1][n2], 0])
            if n1 not in G[n2]:
                attr = G[n2].setdefault(n1, [self.G[n2][n1], 0])
            if k in points:
                attr[-1] += len(points[k])
            if defaultBase:
                attr += [1.0]
            G[n1][n2] = attr
            G[n2][n1] = attr
        return G

    def createProjRandomPointsShp(self, n, out_filename):
        points = nwPtDict = self.getRandomPoints(n, projected=True, toShp=True)
        shp = pysal.open(out_filename, 'w')
        dbf = pysal.open(out_filename[:-3] + 'dbf', 'w')
        dbf.header = ['ID', 'FROM_P1', 'FROM_P2', 'TO_P1', 'TO_P2', 'D_FROM', 'D_TO']
        dbf.field_spec = [('N',9,0)] + [('N',18,7)]*6
        counter = 0
        for k in points:
            for p in points[k]:
                p = list(p)
                shp.write(pysal.cg.Point(tuple(p[-2:])))
                dbf.write([counter, p[0][0], p[0][1], p[1][0], p[1][1], p[2], p[3]])
                counter += 1
        shp.close()
        dbf.close()
            
    def createRandomPointsShp(self, n, out_filename):
        nwPtDict = self.getRandomPoints(n)
        self.writePoints(nwPtDict, out_filename)

    def writePoints(self, points, out_filename):
        shp = pysal.open(out_filename, 'w')
        dbf = pysal.open(out_filename[:-3] + 'dbf', 'w')
        dbf.header = ['ID']
        dbf.field_spec = [('N',9,0)]
        counter = 0
        for k in points:
            for p in points[k]:
                shp.write(pysal.cg.Point(tuple(p)))
                dbf.write([counter])
                counter += 1
        shp.close()
        dbf.close()

    def getClusteredPoints(self, centerNum, ptNum, percent, clusterMeta=False):

        # split network into center- and non-center network
        centerIDs = np.random.randint(0, self.nwNum, centerNum)
        centers = set(centerIDs)
        centerG, centerG_length = {}, 0
        counter, counter2ID = 0, {}
        for center in centerIDs:
            centerG[counter] = self.G[center]
            centerG_length += self.G[center][0]
            counter2ID[counter] = center
            counter += 1
            n1, n2 = self.G[center][1]
            for neighbor in self.GNet[n1]:
                if neighbor != n2:
                    nghLink = self.GNet[n1][neighbor]
                    centerG[counter] = self.G[nghLink]
                    centerG_length += self.G[nghLink][0]
                    counter2ID[counter] = nghLink
                    counter += 1
                    centers.add(nghLink)
            for neighbor in self.GNet[n2]:
                if neighbor != n1:
                    nghLink = self.GNet[n2][neighbor]
                    centerG[counter] = self.G[nghLink]
                    centerG_length += self.G[nghLink][0]
                    counter2ID[counter] = nghLink
                    counter += 1
                    centers.add(nghLink)
        nonCenterIDs = set(self.G.keys()).difference(centers)
        nonCenterG, nonCenterG_length = {}, 0
        for i, nonCenter in enumerate(nonCenterIDs):
             nonCenterG[i] = self.G[nonCenter]
             nonCenterG_length += self.G[nonCenter][0]

        self.oldG, self.old_total_length = self.G, self.total_length

        self.G, self.total_length = centerG, centerG_length
        n_centerPoints = int(percent*ptNum*1.0)
        self.imaginaryLineGenerated = False
        pointsInCenter = self.getRandomPoints(n_centerPoints) 
        meta = {}
        if clusterMeta:
            for cluster in pointsInCenter:
                num_points = len(pointsInCenter[cluster])
                centerLink = self.oldG[cluster][1]
                centerID = counter2ID[cluster]
                centerLink = self.oldG[centerID][1]
                meta[centerLink] = [self.oldG[centerID][0], num_points, centerID in centerIDs]
                #meta[centerID] = [num_points] + list(self.oldG[centerID])
       
        self.G, self.total_length = nonCenterG, nonCenterG_length 
        self.imaginaryLineGenerated = False
        pointsInNonCenter = self.getRandomPoints(ptNum - n_centerPoints)
        centers_no = len(centers)
        for k in pointsInNonCenter:
            pointsInCenter[k + centers_no] = pointsInNonCenter[k]
        pointsInCenter.update(pointsInNonCenter)

        self.G, self.total_length = self.oldG, self.old_total_length

        self.imaginaryLineGenerated = False

        return pointsInCenter, meta

    def writeMeta(self, metaData, out_file):
        shp = pysal.open(out_file, 'w')
        dbf = pysal.open(out_file[:-3] + 'dbf', 'w')
        dbf.header = ['LENGTH', 'NO_PNTS', 'INITIAL_CENTER']
        dbf.field_spec = [('N',9,0)]*2 + [('L',1,0)]
        for link in metaData:
            vertices = list(link)
            vertices = [pysal.cg.Point(v) for v in vertices]
            shp.write(pysal.cg.Chain(vertices))
            dbf.write(metaData[link])
        shp.close()
        dbf.close()

    def createClusteredPointsShp(self, centerNum, n, percent, out_filename, clusterMetaFile=None):
        nwPtDict, meta = self.getClusteredPoints(centerNum, n, percent, clusterMetaFile!=None)
        self.writePoints(nwPtDict, out_filename)
        if clusterMetaFile:
            self.writeMeta(meta, clusterMetaFile)

if __name__ == '__main__':
    sim=Simulation("streets.shp")
    sim.createProjRandomPointsShp(100, "random_100.shp")
    sim.createClusteredPointsShp(2, 100, 0.1, "clustered_100_10p.shp", "clustered_100_10p_meta.shp") 
