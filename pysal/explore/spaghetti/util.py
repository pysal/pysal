from warnings import warn

from pysal.lib import cg
from pysal.lib.common import requires
from rtree import Rtree

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except ImportError:
    err_msg = 'geopandas/shapely not available. '\
              + 'Some functionality will be disabled.'
    warn(err_msg)

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
    
    >>> import pysal.explore.spaghetti as spgh
    >>> point1, point2 = (0,0), (1,1)
    >>> spgh.util.compute_length(point1, point2)
    1.4142135623730951
    
    """
    
    euc_dist = np.sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2)
    
    return euc_dist


def get_neighbor_distances(ntw, v0, l):
    """Get distances to the nearest vertex neighbors along
    connecting arcs.
    
    Parameters
    ----------
    
    ntw : spaghetti.Network
        spaghetti Network object.
    
    v0 : int
        vertex id
    
    l : dict
        key is tuple (start vertex, end vertex); value is ``float``.
        Cost per arc to travel, e.g. distance.
    
    Returns
    -------
    
    neighbors : dict
        key is int (vertex id); value is ``float`` (distance)
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> from pysal.lib import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> neighs = spgh.util.get_neighbor_distances(ntw, 0, ntw.arc_lengths)
    >>> neighs[1]
    102.62353453439829
    
    """
    
    # fetch links associated with vertices
    arcs = ntw.enum_links_vertex(v0)
    
    # create neighbor distance lookup
    neighbors = {}
    
    # iterate over each associated link
    for arc in arcs:
        
        # set distance from vertex1 to vertex2 (link length)
        if arc[0] != v0:
            neighbors[arc[0]] = l[arc]
        else:
            neighbors[arc[1]] = l[arc]
    
    return neighbors


def generatetree(pred):
    """Rebuild the shortest path from root origin to destination.
    
    Parameters
    ----------
    
    pred : list
        List of preceding vertices for traversal route.
    
    Returns
    --------
    
    tree : dict
        key is root origin; value is root origin to destination.
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> from pysal.lib import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, 0)
    >>> tree = spgh.util.generatetree(pred)
    >>> tree[3]
    [23, 22, 20, 19, 170, 2, 0]
    
    """
    
    # instantiate tree lookup
    tree = {}
    
    # iterate over the list of predecessor vertices
    for i, p in enumerate(pred):
        
        # if the route begins/ends with itself set the
        # root vertex and continue to next iteration
        if p == -1:
            
            # tree keyed by root vertex with root vertex as path
            tree[i] = [i]
            continue
        
        # set the initial vertex `p` as `idx`
        idx = p
        # and add it as the first vertex in the path
        path = [idx]
        
        # iterate through the path until back to home vertex
        while idx >= 0:
            # set the next vertex on the path
            next_vertex = pred[idx]
            # and redeclare the current `idx`
            idx = next_vertex
            
            # add the vertex to path while not at home vertex
            if idx >= 0:
                path.append(next_vertex)
        
        # tree keyed by root vertex with network vertices as path
        tree[i] = path
    
    return tree


def dijkstra(ntw, v0, initial_dist=np.inf):
    """Compute the shortest path between a start vertex and
    all other vertices in an origin-destination matrix.
    
    Parameters
    ----------
    
    ntw :  spaghetti.Network
        spaghetti.Network object
    
    v0 : int
        Start vertex ID
    
    initial_dist : float
        Integer break point to stop iteration and return n neighbors.
        Default is ``numpy.inf``.
    
    Returns
    -------
    
    distance : list
        List of distances from vertex to all other vertices.
    
    pred : list
        List of preceeding vertices for traversal route.
    
    Notes
    -----
    
    Based on :cite:`Dijkstra1959a`.
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> from pysal.lib import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, 0)
    >>> round(distance[196], 4)
    5505.6682
    >>> pred[196]
    133
    
    """
    
    # cost per arc to travel, e.g. distance
    cost = ntw.arc_lengths
    
    # initialize travel costs as `inf` for all distances
    distance = [initial_dist for x in ntw.vertex_list]
    
    # label distance to self as 0
    distance[ntw.vertex_list.index(v0)] = 0
    
    # instantiate set of unvisited vertices
    unvisited = set([v0])
    
    # initially label as predecessor vertices with -1 as path
    pred = [-1 for x in ntw.vertex_list]
    
    # iterate over `unvisited` until all vertices have been visited
    while len(unvisited) > 0:
        
        # get vertex with the lowest value from distance
        dist = initial_dist
        
        for vertex in unvisited:
            if distance[vertex] < dist:
                dist = distance[vertex]
                current = vertex
        
        # remove that vertex from the set
        unvisited.remove(current)
        
        # get the neighbors (and costs) to the current vertex
        neighbors = get_neighbor_distances(ntw,
                                           current,
                                           cost)
        
        # iterate over neighbors to find least cost along path
        for v1, indiv_cost in neighbors.items():
            
            # if the labeled cost is greater than
            # the currently calculated cost
            if distance[v1] > distance[current] + indiv_cost:
                
                # relabel to the currently calculated cost
                distance[v1] = distance[current] + indiv_cost
                
                # set the current vertex as a predecessor on the path
                pred[v1] = current
                
                # add the neighbor vertex to `unvisted`
                unvisited.add(v1)
    
    # cast preceding vertices list as an array of integers
    pred = np.array(pred,
                    dtype=np.int)
    
    return distance, pred


def dijkstra_mp(ntw_vertex):
    """Compute the shortest path between a start vertex and all other
    vertices in the web utilizing multiple cores upon request.
    
    Parameters
    ----------
    
    ntw_vertex : tuple
        Tuple of arguments to pass into dijkstra as
        (1) ntw - ``spaghetti.Network object``;
        (2) vertex - ``int``; Start node ID
    
    Returns
    -------
    
    distance : list
        List of distances from vertex to all other vertices.
    
    pred : list
        List of preceeding vertices for traversal route.
    
    Notes
    -----
    
    Based on :cite:`Dijkstra1959a`.
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> from pysal.lib import examples
    >>> ntw = spgh.Network(examples.get_path('streets.shp'))
    >>> distance, pred = spgh.util.dijkstra(ntw, 0)
    >>> round(distance[196], 4)
    5505.6682
    >>> pred[196]
    133
    
    """
    
    # unpack network object and source vertex
    ntw, vertex = ntw_vertex
    
    # calculate shortest path distances and predecessor vertices
    distance, pred = dijkstra(ntw,
                              vertex)
    
    return distance, pred


def squared_distance_point_link(point, link):
    """Find the squared distance between a point and a link.
    
    Parameters
    ----------
    
    point : tuple
        point coordinates (x,y)
    
    link : list
        List of 2 point coordinate tuples [(x0,y0), (x1,y1)].
    
    Returns
    -------
    sqd : float
        distance squared between point and edge
    
    nearp : numpy.ndarray
        array of (xb, yb); the nearest point on the edge
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> point, link = (1,1), ((0,0), (2,0))
    >>> spgh.util.squared_distance_point_link(point, link)
    (1.0, array([1., 0.]))
    
    """
    
    # cast vertices comprising the network link as an array
    p0, p1 = [np.array(p) for p in link]
    
    # cast the observation point as an array
    p = np.array(point)
    
    # subtract point 0 coords from point 1
    v = p1 - p0
    # subtract point 0 coords from the observation coords
    w = p - p0
    
    # if the point 0 vertex is the closest point along the link
    c1 = np.dot(w, v)
    if c1 <= 0.:
        sqd = np.dot(w.T, w)
        nearp = p0
        
        return sqd, nearp
    
    # if the point 1 vertex is the closest point along the link
    c2 = np.dot(v, v)
    if c2 <= c1:
        dp1 = p - p1
        sqd = np.dot(dp1.T, dp1)
        nearp = p1
        
        return sqd, nearp
    
    # otherwise the closest point along the link lies between p0 and p1
    b = c1 / c2
    bv = np.dot(b, v)
    pb = p0 + bv
    d2 = p - pb
    sqd = np.dot(d2, d2)
    nearp = pb
    
    return sqd, nearp


def snap_points_to_links(points, links):
    """Place points onto closest link in a set of links
    (arc/edges)
    
    Parameters
    ----------
    
    points : dict
        Point id as key and (x,y) coordinate as value
    
    links : list
        Elements are of type pysal.lib.cg.shapes.Chain
        ** Note ** each element is a links represented as a chain with
        *one head and one tail vertex* in other words one link only.
    
    Returns
    -------
    
    point2link : dict
        key [point id (see points in arguments)]; value [a 2-tuple 
        ((head, tail), point) where (head, tail) is the target link,
        and point is the snapped location on the link.
    
    Examples
    --------
    
    >>> import pysal.explore.spaghetti as spgh
    >>> from pysal.lib.cg.shapes import Point, Chain
    >>> points = {0: Point((1,1))}
    >>> link = [Chain([Point((0,0)), Point((2,0))])]
    >>> spgh.util.snap_points_to_links(points, link)
    {0: ([(0.0, 0.0), (2.0, 0.0)], array([1., 0.]))}
    
    """
    
    # instantiate an rtree
    rtree = Rtree()
    # set the smallest possible float epsilon on machine
    SMALL = np.finfo(float).eps
    
    # initialize network vertex to link lookup
    vertex_2_link = {}
    
    # iterate over network links
    for i,link in enumerate(links):
        
        # extract network link (x,y) vertex coordinates
        head, tail = link.vertices
        x0, y0 = head
        x1, y1 = tail
        
        if (x0, y0) not in vertex_2_link:
            vertex_2_link[(x0, y0)] = []
        
        if (x1, y1) not in vertex_2_link:
            vertex_2_link[(x1, y1)] = []
        
        vertex_2_link[(x0, y0)].append(link)
        vertex_2_link[(x1, y1)].append(link)
        
        # minimally increase the bounding box exterior
        bx0, by0, bx1, by1 = link.bounding_box
        bx0 -= SMALL
        by0 -= SMALL
        bx1 += SMALL
        by1 += SMALL
        
        # insert the network link and its associated
        # rectangle into the rtree
        rtree.insert(i, (bx0, by0, bx1, by1), obj=link)
        
    # build a KDtree on link vertices
    kdtree = cg.KDTree(list(vertex_2_link.keys()))
    
    point2link = {}
    
    for pt_idx, point in points.items():
        
        # first, find nearest neighbor link vertices for the point
        dmin, vertex = kdtree.query(point, k=1)
        vertex = tuple(kdtree.data[vertex])
        closest = vertex_2_link[vertex][0].vertices
        
        # Use this link as the candidate closest link:  closest
        # Use the distance as the distance to beat:     dmin
        point2link[pt_idx] = (closest, np.array(vertex))
        x0 = point[0] - dmin
        y0 = point[1] - dmin
        x1 = point[0] + dmin
        y1 = point[1] + dmin
        
        # Find all links with bounding boxes that intersect
        # a query rectangle centered on the point with sides
        # of length dmin * dmin
        rtree_lookup = rtree.intersection([x0, y0, x1, y1], objects=True)
        candidates = [cand.object for cand in rtree_lookup]
        dmin += SMALL
        dmin2 = dmin * dmin
        
        # of the candidate arcs, find the nearest to the query point
        for candidate in candidates:
            dist2cand, nearp = squared_distance_point_link(point,
                                                           candidate.vertices)
            if dist2cand <= dmin2:
                closest = candidate.vertices
                dmin2 = dist2cand
                point2link[pt_idx] = (closest,
                                      nearp)
                
    return point2link


@requires('geopandas', 'shapely')
def _points_as_gdf(net, vertices, vertices_for_arcs, pp_name, snapped,
                   id_col=None, geom_col=None):
    """Internal function for returning a point geopandas.GeoDataFrame
    called from within ``spaghetti.element_as_gdf()``.
    
    Parameters
    ----------
    
    vertices_for_arcs : bool
        Flag for points being an object returned [False] or for merely
        creating network arcs [True]. Set from within the parent
        function (``spaghetti.element_as_gdf()``).
    
    Raises
    ------
    
    KeyError
        In order to extract a ``PointPattern`` it must already be a part
        of the ``spaghetti.Network`` object. This exception is raised
        when a ``PointPattern`` is being extracted that does not exist
        within the ``spaghetti.Network`` object.
    
    Returns
    -------
    
    points : geopandas.GeoDataFrame
        Network point elements (either vertices or ``PointPattern``
        points) as a simple ``geopandas.GeoDataFrame`` of
        ``shapely.Point`` objects with an ``id`` column and
        ``geometry`` column.
    
    Notes
    -----
    
    1. See ``spaghetti.element_as_gdf()`` for description of arguments.
    2. This function requires ``geopandas``.
    
    """
    
    # vertices / nodes
    if vertices or vertices_for_arcs:
        pts_dict = net.vertex_coords
    
    # raw point pattern
    if pp_name and not snapped:
        try: 
            pp_pts = net.pointpatterns[pp_name].points
        except KeyError:
            err_msg = 'Available point patterns are {}'
            raise KeyError(err_msg.format(list(net.pointpatterns.keys())))
            
        n_pp_pts = range(len(pp_pts))
        pts_dict = {point:pp_pts[point]['coordinates'] for point in n_pp_pts}
    
    # snapped point pattern
    elif pp_name and snapped:
        pts_dict = net.pointpatterns[pp_name].snapped_coordinates
    
    # instantiate geopandas.GeoDataFrame
    pts_list = list(pts_dict.items())
    points = gpd.GeoDataFrame(pts_list,
                              columns=[id_col,
                                       geom_col])
    points.geometry = points.geometry.apply(lambda p: Point(p))
    
    return points


@requires('geopandas', 'shapely')
def _arcs_as_gdf(net, points, id_col=None, geom_col=None):
    """Internal function for returning a edges geopandas.GeoDataFrame
    called from within ``spaghetti.element_as_gdf()``.
    
    Returns
    -------
    
    points : geopandas.GeoDataFrame
        Network point elements (either vertices or ``PointPattern``
        points) as a simple `geopandas.GeoDataFrame` of
        ``shapely.Point``` objects with an `id` column and
        ``geometry`` column.
    
    Notes
    -----
    
    1. See ``spaghetti.element_as_gdf()`` for description of arguments.
    2. This function requires ``geopandas``.
    
    """
    
    # arcs
    arcs = {}
    
    # iterate over network arcs
    for (vtx1_id, vtx2_id) in net.arcs:
        
        # extract vertices comprising the network arc
        vtx1 = points.loc[(points[id_col] == vtx1_id), geom_col].squeeze()
        vtx2 = points.loc[(points[id_col] == vtx2_id), geom_col].squeeze()
        # create a LineString for the network arc
        arcs[(vtx1_id, vtx2_id)] = LineString((vtx1, vtx2))
    
    # instantiate GeoDataFrame
    arcs = gpd.GeoDataFrame(sorted(list(arcs.items())),
                            columns=[id_col,
                                     geom_col])
    
    # additional columns
    if hasattr(net, 'network_component_labels'):
        arcs['comp_label'] = net.network_component_labels
    
    return arcs
