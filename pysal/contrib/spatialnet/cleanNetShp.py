"""
cleanNetShp -- Tools to clean spatial Network Shapefiles.
"""
import pysal
import numpy

__author__ = "Charles R. Schmidt <schmidtc@gmail.com>"
__all__ = ['snap_verts', 'find_nodes', 'split_at_nodes']


def snap_verts(shp,tolerance=0.001,arc=True):
    """
    snap_verts -- Snap verts that are within tolerance meters of each other.

    Description -- Snapping should be performed with a very small tolerance.
                   The goal is not to change the network, but to ensure rounding
                   errors don't prevent edges from being split at proper intersections.
                   The default of 1mm should be adequate if the input is of decent quality.
                   Higher snapping values can be used to correct digitizing errors, but care
                   should be taken.

    Arguments
    ---------
    tolerance -- float -- snapping tolerance in meters
    arc -- bool -- If true, Ard Distance will be used instead of Euclidean

    Returns
    -------
    generator -- each element is a new pysal.cg.Chain with corrected vertices.
    """
    kmtol = tolerance/1000.

    data = numpy.concatenate([rec.vertices for rec in shp])
    
    if arc:
        kd = pysal.cg.KDTree(data,distance_metric="Arc",radius = pysal.cg.sphere.RADIUS_EARTH_KM)
    else:
        kd = pysal.cg.KDTree(data)
    q = kd.query_ball_tree(kd,kmtol)
    ### Next three lines assert that snappings are mutual... if 1 snaps to 8, 8 must snap to 1.
    for r,a in enumerate(q):
        for o in a:
            assert a==q[o]
    ### non-mutual snapping can happen.
    ### consider the three points, A (-1,0), B (0,0), C (1,0) and a snapping tolerance of 1.
    ### A-> B
    ### B-> A,C
    ### C-> B
    ### For now, try lowering adjusting the tolerance to avoid this.

    data2 = numpy.empty_like(data)
    for i,r in enumerate(q):
        data2[i] = data[r].mean(0)
    pos=0
    for rec in shp:
        vrts = rec.vertices
        n = len(vrts)
        nrec = pysal.cg.Chain(map(tuple,data2[pos:pos+n]))
        pos+=n
        yield nrec
    
def find_nodes(shp):
    """
    find_nodes -- Finds vertices in a line type shapefile that appear more than once and/or are end points of a line

    Arguments
    ---------
    shp -- Shapefile Object -- Should be of type Line.

    Returns
    -------
    set
    """
    node_count = {}
    for road in shp:
        vrts = road.vertices
        for node in vrts:
            if node not in node_count:
                node_count[node] = 0
            node_count[node] += 1
        node_count[vrts[0]] += 1
        node_count[vrts[-1]] += 1
    return set([node for node,c in node_count.iteritems() if c > 1])

def split_at_nodes(shp):
    """
    split_at_nodes -- Split line features at nodes

    Arguments
    ---------
    shp -- list or shapefile -- Chain features to be split at common nodes.

    Returns
    -------
    generator -- yields pysal.cg.Chain objects
    """
    nodes = find_nodes(shp)
    nodeIds = list(nodes)
    nodeIds.sort()
    nodeIds = dict([(node,i) for i,node in enumerate(nodeIds)])
    
    for road in shp:
        vrts = road.vertices
        midVrts = set(road.vertices[1:-1]) #we know end points are nodes
        midNodes = midVrts.intersection(nodes) # find any nodes in the middle of the feature.
        midIdx = [vrts.index(node) for node in midNodes] # Get their indices
        midIdx.sort()
        if midIdx:
            #print vrts
            starts = [0]+midIdx
            stops = [x+1 for x in midIdx]+[None]
            for start,stop in zip(starts,stops):
                feat = pysal.cg.Chain(vrts[start:stop])
                rec = (nodeIds[feat.vertices[0]],nodeIds[feat.vertices[-1]],False)
                yield feat,rec
        else:
            rec = (nodeIds[road.vertices[0]],nodeIds[road.vertices[-1]],False)
            yield road,rec


def createSpatialNetworkShapefile(inshp,outshp):
    assert inshp.lower().endswith('.shp')
    assert outshp.lower().endswith('.shp')
    shp = pysal.open(inshp,'r')
    snapped = list(snap_verts(shp,.001))
    o = pysal.open(outshp,'w')
    odb = pysal.open(outshp[:-4]+'.dbf','w')
    odb.header = ["FNODE","TNODE","ONEWAY"]
    odb.field_spec = [('N',20,0),('N',20,0),('L',1,0)]

    new = list(split_at_nodes(snapped))
    for feat,rec in new:
        o.write(feat)
        odb.write(rec)
    o.close()
    odb.close()
    print "Split %d roads in %d network edges"%(len(shp),len(new))

if __name__=='__main__':
    createSpatialNetworkShapefile('beth_roads.shp','beth_network.shp')

