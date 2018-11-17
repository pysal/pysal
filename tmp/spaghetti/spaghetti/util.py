from libpysal import cg
import numpy as np


def compute_length(v0, v1):
    """Compute the euclidean distance between two points.
    
    Parameters
    ----------
    
    v0 : tuple
        sequence in the form x, y
    
    vq : tuple
        sequence in the form x, y
    
    Returns
    --------
    
    euc_dist : float
        Euclidean distance
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> point1, point2 = (0,0), (1,1)
    >>> spgh.util.compute_length(point1, point2)
    1.4142135623730951
    """
    euc_dist = np.sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2)
    return euc_dist


def get_neighbor_distances(ntw, v0, l):
    """Get distances to the nearest node neighbors along connecting edges.
    
    Parameters
    ----------
    ntw : spaghetti.Network
        spaghetti Network object.
    v0 : int
        Node id
    l : dict
        key is tuple (start node, end node); value is float.
        Cost per edge to travel, e.g. distance.
    
    Returns
    -------
    neighbors : dict
        key is int (node id); value is float (distance)
    
    Examples
    --------
    >>> import spaghetti as spgh
    >>> from libpysal import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> neighs = spgh.util.get_neighbor_distances(ntw, 0, ntw.edge_lengths)
    >>> neighs[1]
    102.62353453439829
    """
    edges = ntw.enum_links_node(v0)
    neighbors = {}
    for e in edges:
        if e[0] != v0:
            neighbors[e[0]] = l[e]
        else:
            neighbors[e[1]] = l[e]
    return neighbors


def generatetree(pred):
    """Rebuild the shortest path from root origin to destination
    
    Parameters
    ----------
    
    pred : list
        List of preceeding nodes for traversal route.
    
    Returns
    --------
    
    tree : dict
        key is root origin; value is root origin to destination.
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> from libpysal import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, ntw.edge_lengths, 0)
    >>> tree = spgh.util.generatetree(pred)
    >>> tree[3]
    [23, 22, 20, 19, 170, 2, 0]
    """
    tree = {}
    for i, p in enumerate(pred):
        if p == -1:
            # root node
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


def dijkstra(ntw, cost, v0, n=float('inf')):
    """Compute the shortest path between a start node and all other nodes in
    an origin-destination matrix.

    Parameters
    ----------
    
    ntw :  spaghetti.Network
        spaghetti Network object.
    
    cost : dict
        key is tuple (start node, end node); value is float.
        Cost per edge to travel, e.g. distance.
    
    v0 : int
        Start node ID
    
    n : float
        integer break point to stop iteration and return n neighbors.
        Default is ('inf').
    
    Returns
    -------
    
    distance : list
        List of distances from node to all other nodes.
    
    pred : list
        List of preceeding nodes for traversal route.
    
    Notes
    -----
    
    Based on :cite:`Dijkstra1959a`.
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> from libpysal import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, ntw.edge_lengths, 0)
    >>> round(distance[196], 4)
    5505.6682
    >>> pred[196]
    133
    """
    distance = [n for x in ntw.node_list]
    idx = ntw.node_list.index(v0)
    distance[ntw.node_list.index(v0)] = 0
    pred = [-1 for x in ntw.node_list]
    a = set()
    a.add(v0)
    while len(a) > 0:
        # Get node with the lowest value from distance.
        dist = n
        for node in a:
            if distance[node] < dist:
                dist = distance[node]
                v = node
        # Remove that node from the set.
        a.remove(v)
        last = v
        # 4. Get the neighbors to the current node.
        neighbors = get_neighbor_distances(ntw, v, cost)
        for v1, indiv_cost in neighbors.items():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
    pred = np.array(pred, dtype=np.int)
    return distance, pred


def dijkstra_mp(ntw_cost_node):
    """
    Compute the shortest path between a start node and all other
    nodes in the web utilizing multiple cores upon request.
    
    Parameters
    ----------
    
    ntw_cost_node : tuple
        tuple of arguments to pass into dijkstra
        (1) ntw - spaghetti.Network; spaghetti Network object; (2) cost - dict;
        key is tuple (start node, end node); value is float - Cost per edge to
        travel, e.g. distance; (3) node - int; Start node ID
    
    Returns
    -------
    
    distance : list
        List of distances from node to all other nodes.
    
    pred : list
        List of preceeding nodes for traversal route.
    
    Notes
    -----
    
    Based on :cite:`Dijkstra1959a`.
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> from libpysal import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, ntw.edge_lengths, 0)
    >>> round(distance[196], 4)
    5505.6682
    >>> pred[196]
    133
    """
    ntw, cost, node = ntw_cost_node
    distance, pred = dijkstra(ntw, cost, node)
    return distance, pred


def squared_distance_point_segment(point, segment):
    """Find the squared distance between a point and a segment.
    
    Parameters
    ----------
    
    point : tuple
        point coordinates (x,y)
    
    segment : list
        List of 2 point coordinate tuples [(x0,y0), (x1,y1)].
    
    Returns
    -------
    sqd : float
        distance squared between point and segment
    
    nearp : numpy.ndarray
        array of (xb, yb); the nearest point on the segment
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> point, segment = (1,1), ((0,0), (2,0))
    >>> spgh.util.squared_distance_point_segment(point, segment)
    (1.0, array([1., 0.]))
    """
    #
    p0, p1 = [np.array(p) for p in segment]
    v = p1 - p0
    p = np.array(point)
    w = p - p0
    c1 = np.dot(w, v)
    if c1 <= 0.:
        sqd = np.dot(w.T, w)
        nearp = p0
        return sqd, nearp
    #
    c2 = np.dot(v, v)
    if c2 <= c1:
        dp1 = p - p1
        sqd = np.dot(dp1.T, dp1)
        nearp = p1
        return sqd, nearp
    #
    b = c1 / c2
    bv = np.dot(b, v)
    pb = p0 + bv
    d2 = p - pb
    sqd = np.dot(d2, d2)
    nearp = pb
    return sqd, nearp


def snap_points_on_segments(points, segments):
    """Place points onto closet segment in a set of segments
    
    Parameters
    ----------
    
    points : dict
        Point id as key and (x,y) coordinate as value
    
    segments : list
        Elements are of type libpysal.cg.shapes.Chain
        ** Note ** each element is a segment represented as a chain with
        *one head and one tail node* in other words one link only.
    
    Returns
    -------
    
    p2s : dict
        key [point id (see points in arguments)]; value [a 2-tuple 
        ((head, tail), point) where (head, tail) is the target segment,
        and point is the snapped location on the segment.
    
    Examples
    --------
    
    >>> import spaghetti as spgh
    >>> from libpysal.cg.shapes import Point, Chain
    >>> points = {0: Point((1,1))}
    >>> segments = [Chain([Point((0,0)), Point((2,0))])]
    >>> spgh.util.snap_points_on_segments(points, segments)
    {0: ([(0.0, 0.0), (2.0, 0.0)], array([1., 0.]))}
    """
    
    # Put segments in an Rtree.
    rt = cg.Rtree()
    SMALL = np.finfo(float).eps
    node2segs = {}
    
    for segment in segments:
        head, tail = segment.vertices
        x0, y0 = head
        x1, y1 = tail
        if (x0, y0) not in node2segs:
            node2segs[(x0, y0)] = []
        if (x1, y1) not in node2segs:
            node2segs[(x1, y1)] = []
        node2segs[(x0, y0)].append(segment)
        node2segs[(x1, y1)].append(segment)
        x0, y0, x1, y1 = segment.bounding_box
        x0 -= SMALL
        y0 -= SMALL
        x1 += SMALL
        y1 += SMALL
        r = cg.Rect(x0, y0, x1, y1)
        rt.insert(segment, r)
        
    # Build a KDtree on segment nodes.
    kt = cg.KDTree(list(node2segs.keys()))
    p2s = {}
    
    for ptIdx, point in points.items():
        # First, find nearest neighbor segment node for the point.
        dmin, node = kt.query(point, k=1)
        node = tuple(kt.data[node])
        closest = node2segs[node][0].vertices
        
        # Use this segment as the candidate closest segment:  closest
        # Use the distance as the distance to beat:           dmin
        p2s[ptIdx] = (closest, np.array(node))
        x0 = point[0] - dmin
        y0 = point[1] - dmin
        x1 = point[0] + dmin
        y1 = point[1] + dmin
        
        # Find all segments with bounding boxes that intersect
        # a query rectangle centered on the point with sides of length 2*dmin.
        candidates = [cand for cand in rt.intersection([x0, y0, x1, y1])]
        dmin += SMALL
        dmin2 = dmin * dmin
        
        # Of the candidate segments, find the nearest to the query point.
        for candidate in candidates:
            dnc, p2b = squared_distance_point_segment(point,
                                                      candidate.vertices)
            if dnc <= dmin2:
                closest = candidate.vertices
                dmin2 = dnc
                p2s[ptIdx] = (closest, p2b)
                
    return p2s