from collections import OrderedDict
import math
import operator
import pysal as ps
import numpy as np



def compute_length(v0, v1):
    """
    Compute the euclidean distance between two points.

    Parameters
    ----------
    v0      sequence in the form x, y
    vq      sequence in the form x, y

    Returns
    --------
    Euclidean distance
    """

    return math.sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2)


def get_neighbor_distances(ntw, v0, l):
    edges = ntw.enum_links_node(v0)
    neighbors = {}
    for e in edges:
        if e[0] != v0:
            neighbors[e[0]] = l[e]
        else:
            neighbors[e[1]] = l[e]
    return neighbors


def generatetree(pred):
    tree = {}
    for i, p in enumerate(pred):
        if p == -1:
            #root node
            tree[i] = [i]
            continue
        idx = p
        path = [idx]
        while idx >= 0:
            nextnode = pred[idx]
            idx = nextnode
            if idx >= 0:
                path.append(nextnode)
        tree[i] = path
    return tree

def dijkstra(ntw, cost, node, n=float('inf')):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.
    Parameters
    ----------
    ntw: PySAL network object
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID
    n: integer break point to stop iteration and return n
     neighbors
    Returns:
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    v0 = node
    distance = [float('inf') for x in ntw.node_list]
    idx = ntw.node_list.index(v0)
    distance[ntw.node_list.index(v0)] = 0
    pred = [-1 for x in ntw.node_list]
    a = set()
    a.add(v0)
    while len(a) > 0:
        #Get node with the lowest value from distance
        dist = float('inf')
        for node in a:
            if distance[node] < dist:
                dist = distance[node]
                v = node
        #Remove that node from the set
        a.remove(v)
        last = v
        #4. Get the neighbors to the current node
        neighbors = get_neighbor_distances(ntw, v, cost)
        for v1, indiv_cost in neighbors.iteritems():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
    return distance, np.array(pred, dtype=np.int)


def squaredDistancePointSegment(point, segment):
    """Find the squared distance between a point and a segment
    
    Arguments
    =========
    
    point: tuple (x,y)
    
    segment: list of 2 tuples [(x0,y0), (x1,y1)]
    
    Returns
    =======
    
    tuple: 2 elements
    
           distance squared between point and segment
    
           array(xb, yb): the nearest point on the segment
    
    """
    p0,p1 = [np.array(p) for p in segment]
    v = p1 - p0
    p = np.array(point)
    w = p - p0
    c1 = np.dot(w,v)
    if c1 <= 0.:
        # print 'before p0'
        return np.dot(w.T,w), p0
    c2 = np.dot(v,v)
    if c2 <= c1:
        dp1 = p - p1
        # print 'after p1'
        return np.dot(dp1.T,dp1), p1
    
    b = c1 / c2
    bv = np.dot(b,v)
    pb = p0 + bv
    d2 = p - pb
    
    return np.dot(d2,d2), pb
    


def snapPointsOnSegments(points, segments):
    """Place points onto closet segment in a set of segments
    
    Arguments
    =========
    
    points: dict
            with point id as key and (x,y) coordinate as value
    
    segments: list
              elements are of type pysal.cg.shapes.Chain 
              Note that the each element is a segment represented as a chain with *one head and one tail node*, in other words one link only.
              
    Returns
    =======
    
    p2s: dictionary
         key:  point id (see points in arguments)
         
         value:  a 2-tuple: ((head, tail), point)
                 where (head, tail) is the target segment, and point is the snapped location on the segment
              
    """
    
    # Put segments in an Rtree
    rt = ps.cg.Rtree()
    SMALL = 0.01
    node2segs = {}
    
    for segment in segments:
        head,tail = segment.vertices
        x0,y0 = head
        x1,y1 = tail
        if (x0,y0) not in node2segs:
            node2segs[(x0,y0)] = []
        if (x1,y1) not in node2segs:
            node2segs[(x1,y1)] = []
        node2segs[(x0,y0)].append(segment)
        node2segs[(x1,y1)].append(segment)
        x0,y0,x1,y1 =  segment.bounding_box
        x0 -= SMALL
        y0 -= SMALL
        x1 += SMALL
        y1 += SMALL
        r = ps.cg.Rect(x0,y0,x1,y1)
        rt.insert(segment, r)
        
        
        
    # Build a KDtree on segment nodes
    kt = ps.cg.KDTree(node2segs.keys())
    p2s = {}

    for ptIdx, point in points.iteritems():
        # first find nearest neighbor segment node for point
        dmin, node = kt.query(point, k=1)
        node = tuple(kt.data[node])
        closest = node2segs[node][0].vertices
        
        # use this segment as the candidate closest segment: closest
        # use the distance as the distance to beat: dmin
        p2s[ptIdx] = (closest, node) # sna
        x0 = point[0] - dmin
        y0 = point[1] - dmin
        x1 = point[0] + dmin
        y1 = point[1] + dmin
        
        # find all segments with bounding boxes that intersect
        # a query rectangle centered on the point with sides of length 2*dmin
        candidates = [ cand for cand in rt.intersection([x0,y0,x1,y1])]
        dmin += SMALL
        dmin2 = dmin * dmin
        
        # of the candidate segments, find the one that is the minimum distance to the query point
        for candidate in candidates:
            dnc, p2b = squaredDistancePointSegment(point, candidate.vertices)
            if dnc <= dmin2:
                closest = candidate.vertices
                dmin2 = dnc
                p2s[ptIdx] = (closest, p2b)
        
    return p2s
    
