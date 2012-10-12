import pysal
from pysal.cg.segmentLocator import Polyline_Shapefile_SegmentLocator
import networkx

EUCLIDEAN_DISTANCE = "Euclidean"
ARC_DISTANCE = "Arc"

class SpatialNetwork(object):
    """
    SpatialNetwork -- Represents a Spatial Network. 

    A Spatial Network in PySAL is a graph who's nodes and edges are represented by geographic features 
    such as Points for nodes and Lines for edges.
    An example of a spatial network is a road network.

    Arguments
    ---------
    shapefile -- Shapefile contains the geographic represention of the network.
                 The shapefile must be a polyline type and the associated DBF MUST contain the following fields:
                    FNODE -- source node -- ID of the source node, the first vertex in the polyline feature.
                    TNODE -- destination node -- ID of the destination node, the last vertex in the polyline feature.
                    ONEWAY -- bool -- If True, the edge will be marked oneway starting at FNODE and ending at TNODE
    distance_metric -- EUCLIDEAN_DISTANCE or ARC_DISTANCE
    
    """
    def __init__(self,shapefile,distance_metric=EUCLIDEAN_DISTANCE):
        if issubclass(type(shapefile),basestring): #Path
            self.shp = shp = pysal.open(shapefile,'r')
        else:
            raise TypeError,"Expecting a string, shapefile should the path to shapefile"
        if shp.type != pysal.cg.shapes.Chain:
            raise ValueError,"Shapefile must contain polyline features"
        self.dbf = dbf = pysal.open(shapefile[:-4]+'.dbf','r')
        header = dbf.header
        if (('FNODE' not in header) or ('TNODE' not in header) or ('ONEWAY' not in header)):
            raise ValueError,"DBF must contain: FNODE,TNODE,ONEWAY"
        
        oneway = [{'F':False,'T':True}[x] for x in dbf.by_col('ONEWAY')]
        fnode = dbf.by_col('FNODE')
        tnode = dbf.by_col('TNODE')
        if distance_metric == EUCLIDEAN_DISTANCE:
            lengths = [x.len for x in shp]
        elif distance_metric == ARC_DISTANCE:
            lengths = [x.arclen for x in shp]
        else:
            raise ValueError,"distance_metric must be either EUCLIDEAN_DISTANCE or ARC_DISTANCE"
        self.lengths = lengths
        if any(oneway):
            G = networkx.MultiDiGraph()
            #def isoneway(x):
            #    return x[-1]
            #def isnotoneway(x):
            #    if(x[-1]):
            #        return False
            #    return True
            #    
            #A = filter(isoneway,zip(fnode,tnode,oneway))
            #B = filter(isnotoneway,zip(fnode,tnode,oneway))
            #C = filter(isnotoneway,zip(tnode,fnode,oneway))
            #G.add_edges_from(A)
            #G.add_edges_from(B)
            #G.add_edges_from(C)
        else:
            G = networkx.MultiGraph()
        #zip(fnode,tnode))
        self.G = G
        shp.seek(0)
        self._locator = Polyline_Shapefile_SegmentLocator(shp)
    def snap(self,pt):
        i,p,j = self._locator.nearest(pt) #shpID,partID,segmentID
        segment = self.shp[i].segments[p][j] #grab segment
        d,pct = pysal.cg.get_segment_point_dist(segment,pt) #find pct along segment
        x0,x1 = segment.p1[0],segment.p2[0]
        x2 = x0 + (x1-x0)*pct # find x location of snap
        y2 = segment.line.y(x2) # find y location of snap

        #dbf = self.dbf
        #rec = dict(zip(dbf.header,dbf[i][0]))
        #edge = (rec['FNODE'],rec['TNODE'])
        #TODO: Calculate location along edge and distance to edge"
        #return edge
        return x2,y2
        
    
if __name__=='__main__':
    import random
    net = SpatialNetwork('beth_network.shp',ARC_DISTANCE)

    n = 1000
    minX,minY,maxX,maxY = net.shp.bbox
    xRange = maxX-minX
    yRange = maxY-minY
    qpts = [(random.random(), random.random()) for i in xrange(n)]
    qpts = [pysal.cg.Point((minX+(xRange*x),minY+(yRange*y))) for x,y in qpts]
    o = pysal.open('random_qpts.shp','w')
    for p in qpts:
        o.write(p)
    o.close()
    o = pysal.open('random_qpts_snapped.shp','w')
    for qpt in qpts:
        spt = net.snap(qpt)
        o.write(pysal.cg.Chain([qpt,spt]))
    o.close()
    



