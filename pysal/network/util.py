"""
Utilities for PySAL Network Module
"""

__author__ = "Sergio Rey <sjsrey@gmail.com>, Jay Laura <jlaura@asu.edu>"
import copy
import math
from collections import OrderedDict
from random import uniform
import multiprocessing as mp
from heapq import nsmallest

import pysal as ps
import numpy as np
from pysal.cg.standalone import get_points_dist


class SortedEdges(OrderedDict):
    def next_key(self, key):
        next = self._OrderedDict__map[key][1]
        if next is self._OrderedDict__root:
            raise ValueError("{!r} is the last key.".format(key))
        return next[2]
    def first_key(self):
        for key in self: return key
        raise ValueError("No sorted edges remain.")

def enum_links_node(wed, node):
    """
    Enumerate links in cw order around a node

    Parameters
    ----------

    node: string/int
        id for the node in wed


    Returns
    -------

    links: list
        links ordered cw around node
    """

    links = []
    if node not in wed.node_edge:
        return links
    l0 = wed.node_edge[node]
    links.append(l0)
    l = l0
    v = node
    searching = True
    while searching:
        if v == l[0]:
            l = wed.start_c[l]
        else:
            l = wed.end_c[l]
        if (l is None) or (set(l) == set(l0)):
            searching = False
        else:
            links.append(l)
    return links


def enum_edges_region(wed, region):
    """
    Enumerate the edges of a region/polygon in cw order

    Parameters
    ----------

    region: id for the region in wed


    Returns
    -------

    links: list of links ordered cw that define the region/polygon

    """
    right_polygon = wed.right_polygon
    end_cc = wed.end_cc
    start_cc = wed.start_cc
    region_edge = wed.region_edge
    l0 = region_edge[region]
    l = copy.copy(l0)
    edges = []
    edges.append(l)
    traveling = True
    while traveling:
        if region == right_polygon[l]:
            l = end_cc[l]
        else:
            l = start_cc[l]
        edges.append(l)
        if set(l) == set(l0):
            traveling = False
    return edges


def edge_length(wed, half=False):
    """
    Compute the cartesian length of all edges.  This is a helper
        function to allow for ratio data with spatial autocorrelation
        analysis.

    Parameters
    ----------
    wed: PySAL Winged Edged Data structure
    half: Double edge length by default, flag to set to single

    Returns
    -------
    length : dict {tuple(edge): float(length)}
        The length of each edge.
    """

    lengths = {}
    for edge in wed.edge_list:
        if half:
            if (edge[1], edge[0]) not in lengths.keys():
                lengths[edge] = get_points_dist(wed.node_coords[edge[0]],
                                                wed.node_coords[edge[1]])
        else:
            lengths[edge] = get_points_dist(wed.node_coords[edge[0]],
                                            wed.node_coords[edge[1]])
    return lengths


def snap_to_edges(wed, points):
    """
    Snaps observations to the netwrok edge.

    Parameters
    wed: PySAL Winged Edged Data Structure

    Returns:
    --------
    obs_to_edge: a dict of dicts {edge:{point_id:(x,y)}}
    """

    obs_to_edge = {}
    for region in wed.region_edge.keys():
        verts = []
        region_edges = enum_edges_region(wed, region)
        for edge in region_edges:
            if edge[0] not in verts:
                verts.append(edge[0])
            elif edge[1] not in verts:
                verts.append(edge[1])
        verts.append(verts[0])
        poly = ps.cg.Polygon([wed.node_coords[v] for v in verts])

        for pt_index, point in enumerate(points):
            x0 = point[0]
            y0 = point[1]
            if ps.cg.standalone.get_polygon_point_intersect(poly, point):
                d = {}
                vectors = {}
                c = 0
                ccw_edges = region_edges[::-1]
                for i in range(len(ccw_edges)-1):
                    edge = ccw_edges[i]
                    xi = wed.node_coords[edge[0]][0]
                    yi = wed.node_coords[edge[0]][1]
                    xi1 = wed.node_coords[edge[1]][0]
                    yi1 = wed.node_coords[edge[1]][1]
                    num = ((yi1 - yi)*(x0-xi)-(xi1-xi)*(y0-yi))
                    denom = ((yi1-yi)**2 + (xi1-xi)**2)
                    #num = abs(((xi - xi1) * (y0 - yi)) - ((yi - yi1) * (x0 - xi)))
                    #denom = math.sqrt(((xi - xi1)**2) + (yi-yi1)**2)
                    k = num / denom
                    distance = abs(num) / math.sqrt(((yi1-yi)**2 + (xi1-xi)**2))
                    vectors[c] = (xi, xi1, yi, yi1,k,edge)
                    d[distance] = c
                    c += 1
                min_dist = SortedEdges(sorted(d.items()))
                for dist, vector_id in min_dist.iteritems():
                    value = vectors[vector_id]
                    xi = value[0]
                    xi1 = value[1]
                    yi = value[2]
                    yi1 = value[3]
                    k = value[4]
                    edge = value[5]
                    #Okabe Method
                    x = x0 - k * (yi1 - yi)
                    y = y0 + k * (xi1 - xi)
                    if xi <= x <= xi1 or xi1 <= x <= xi and yi <= y <= yi1 or yi1 <=y <= yi:
                        #print "{} intersections edge {} at {}".format(pt_index, edge, (x,y))
                        if edge not in obs_to_edge.keys():
                            obs_to_edge[edge] = {pt_index:(x,y)}
                        else:
                            obs_to_edge[edge][pt_index] = (x,y)
                        break
                    else:
                        #either pi or pi+1 are the nearest point on that edge.
                        #If this point is closer than the next distance, we can break, the
                        # observation intersects the node with the shorter
                        # distance.
                        pi = (xi, yi)
                        pi1 = (xi1, yi1)
                        p0 = (x0,y0)
                        dist_pi = ps.cg.standalone.get_points_dist(p0, pi)
                        dist_pi1 = ps.cg.standalone.get_points_dist(p0, pi1)
                        if dist_pi < dist_pi1:
                            node_dist = dist_pi
                            (x,y) = pi
                        else:
                            node_dist = dist_pi1
                            (x,y) = pi1
                        if node_dist < min_dist.next_key(dist):
                            if edge not in obs_to_edge.keys():
                                obs_to_edge[edge] = {pt_index:(x,y)}
                            else:
                                obs_to_edge[edge][pt_index] = (x,y)
                            break

    return obs_to_edge


def count_per_edge(obs_on_network):
    """
    Snaps observations to the nearest edge and then counts
        the number of observations per edge.

    Parameters
    ----------
    obs_on_network: dict of observations on the network
        {(edge): {pt_id: (coords)}} or {edge: [(coord), (coord), (coord)]}
    Returns
    -------
    counts: dict {(edge):count}
    """

    counts = {}
    for key in obs_on_network.iterkeys():
        counts[key] = len(obs_on_network[key])
    return counts


def simulate_observations(wed, count, distribution='uniform'):
    """
    Generates simulated points to test for NCSR

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    count: integer number of points to create
    distirbution: distribution of random points

    Returns
    -------
    random_pts: dict with {(edge):[(x,y), (x1, y1), ... , (xn,yn)]}
    """
    if distribution is 'uniform':
        lengths = wed.edge_length()
        single_lengths = []
        edges = []
        for edge, l in lengths.iteritems():
            if (edge[0], edge[1]) or (edge[1], edge[0]) not in edges:
                edges.append(edge)
                single_lengths.append(l)
        line = np.array([single_lengths])
        offsets = np.cumsum(line)
        total_length = offsets[-1]
        starts = np.concatenate((np.array([0]), offsets[:-1]), axis=0)
        random_pts = {}
        for x in range(count):
            random_pt = uniform(0,total_length)
            start_index = np.where(starts <= random_pt)[0][-1]
            assignment_edge = edges[start_index]
            distance_from_start = random_pt - offsets[start_index - 1]
            x0, y0 = newpoint_coords(wed, assignment_edge, distance_from_start)
            if assignment_edge not in random_pts.keys():
                random_pts[assignment_edge] = [(x0,y0)]
            else:
                random_pts[assignment_edge].append((x0,y0))
    return random_pts


def newpoint_coords(wed, edge, distance):
    x1 = wed.node_coords[edge[0]][0]
    y1 = wed.node_coords[edge[0]][1]
    x2 = wed.node_coords[edge[1]][0]
    y2 = wed.node_coords[edge[1]][1]
    m = (y2 - y1) / (x2 - x1)
    b1 = y1 - m * (x1)
    if x1 > x2:
        x0 = x1 - distance / math.sqrt(1 + m**2)
    elif x1 < x2:
        x0 = x1 + distance / math.sqrt(1 + m**2)
    y0 = m * (x0 - x1) + y1
    return x0, y0


def dijkstra(wed, cost, node, n=float('inf')):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID
    n: integer break point to stop iteration and return n
     neighbors

    Returns:
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    v0 = node
    distance = [float('inf') for x in wed.node_list]
    distance[wed.node_list.index(v0)] = 0
    pred = [None for x in wed.node_list]
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
        neighbors = get_neighbor_distances(wed, v, cost)
        for v1, indiv_cost in neighbors.iteritems():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
    return distance, pred


def shortest_path(wed, cost, start, end):
    distance, pred = dijkstra(wed, cost, start)
    path = [end]
    previous = pred[end]
    while previous != start:
        path.append(previous)
        end = previous
        previous = pred[end]
    path.append(start)
    return path


def check_connectivity(wed, cost):
    distance, pred = dijkstra(wed, cost, 0)
    if float('inf') in distance:
        return False
    else:
        return True


def get_neighbor_distances(wed, v0, l):
    edges = wed.enum_links_node(v0)
    neighbors = {}
    for e in edges:
        if e[0] != v0:
            neighbors[e[0]] = l[e]
        else:
            neighbors[e[1]] = l[e]
    return neighbors

def newpointer(k,v,c):
    '''
    Helper function for node insertion.

    Parameters
    ----------
    k: key of the pointer to update
    v: current value of the key
    c: value to replace the non-shared (k:v) value with

    Returns
    -------
    l: tuple new value of the pointer k
    '''
    mask = [(i == j) for i, j in zip(k, v)]
    if all(x == False for x in mask):
        mask = [(i == j) for i, j in zip(k, (v[1], v[0]))]
    replace = mask.index(False)
    l = list(v)
    l[replace] = c
    return tuple(l)


def insert_node(wed, edge, distance, segment=False):
    """
    Insert a node into an existing edge at a fixed distance from
     the start node.

    Parameters
    ----------
    wed: PySAL Winged Edge Data Structure
    edge: The edge to insert a node into
    distance: float, distance from the start node of the edge
    segment: a flag that returns the modified WED and the new node id

    Returns
    -------
    wed: Modified PySAL WED data structure
    """
    #Get the coordinates of the new point and update the node_coords
    x0, y0 = newpoint_coords(wed, edge, distance)
    newcoord_id = max(wed.node_list) + 1
    wed.node_list.append(newcoord_id)
    wed.node_coords[newcoord_id] = (x0, y0)
    #Update the region edge
    new_edge = (edge[0], newcoord_id)
    if edge in wed.region_edge.keys():
        wed.region_edge[new_edge] = wed.region_edge.pop(edge)
    if (edge[1], edge[0]) in wed.region_edge.keys():
        wed.region_edge[(new_edge[1], new_edge[0])] = wed.region_edge.pop((edge[1], edge[0]))

    a = edge[0]
    b = edge[1]
    c = newcoord_id
    #Update the edge list
    idx = wed.edge_list.index(edge)
    wed.edge_list.pop(idx)
    wed.edge_list += [(a, c), (c, a)]
    idx = wed.edge_list.index((b, a))
    wed.edge_list.pop(idx)
    wed.edge_list += [(b, c), (c, b)]
    #Update the start and end nodes
        #Remove the old start and end node pointers
    wed.start_node.pop(edge)
    wed.start_node.pop((b, a))
    wed.end_node.pop(edge)
    wed.end_node.pop((b, a))
        #Add the 4 new pointers
    wed.start_node[(a, c)] = a
    wed.end_node[(a, c)] = c
    wed.start_node[(c, a)] = c
    wed.end_node[(c, a)] = a
    wed.start_node[(c, b)] = c
    wed.end_node[(c, b)] = b
    wed.start_node[(b, c)] = b
    wed.end_node[(b, c)] = c
    #Update the startc, startcc, enc, endcc of the new links
        #Replace the old pointers with new pointers
    wed.start_c[(a, c)] = wed.start_c.pop(edge)
    wed.start_cc[(a, c)] = wed.start_cc.pop(edge)
    wed.end_c[(c, b)] = wed.end_c.pop(edge)
    wed.end_cc[(c, b)] = wed.end_cc.pop(edge)
    rev_edge = (b, a)
    wed.start_c[(b, c)] = wed.start_c.pop(rev_edge)
    wed.start_cc[(b, c)] = wed.start_cc.pop(rev_edge)
    wed.end_c[(c, a)] = wed.end_c.pop(rev_edge)
    wed.end_cc[(c, a)] = wed.end_cc.pop(rev_edge)
        #Add brand new pointers for the new edges
    wed.start_c[(c, a)] = (c, b)
    wed.start_cc[(c, a)] = (c, b)
    wed.end_c[(a, c)] = (c, b)
    wed.end_cc[(a, c)] = (c, b)
    wed.start_c[(c, b)] = (c, a)
    wed.start_cc[(c, b)] = (c, a)
    wed.end_c[(b, c)] = (c, a)
    wed.end_cc[(b, c)] = (c, a)
    #Update the pointer to the nodes incident to start / end of the original link
    for k, v in wed.start_c.iteritems():
        if v == edge:
            wed.start_c[k] = newpointer(k,v,c)
        elif v == rev_edge:
            wed.start_c[k] = newpointer(k,v,c)
    for k, v in wed.start_cc.iteritems():
        if v == edge:
            wed.start_cc[k] = newpointer(k,v,c)
        elif v == rev_edge:
            wed.start_cc[k] = newpointer(k,v,c)
    for k, v in wed.end_c.iteritems():
        if v == edge:
            wed.end_c[k] = newpointer(k,v,c)
        elif v == rev_edge:
            wed.end_c[k] = newpointer(k,v,c)
    for k, v in wed.end_cc.iteritems():
        if v == edge:
            wed.end_cc[k] = newpointer(k,v,c)
        elif v == rev_edge:
            wed.end_cc[k] = newpointer(k,v,c)
    #Update the node_edge pointer
    wed.node_edge[c] = (a, c)
    wed.node_edge[a] = (a, c)
    wed.node_edge[b] = (b, c)
    #update right and left polygon regions
    if edge in wed.right_polygon.keys():
        right = wed.right_polygon.pop(edge)
        wed.right_polygon[(a, c)] = right
        wed.right_polygon[(c, b)] = right
    if edge in wed.left_polygon.keys():
        left = wed.left_polygon.pop(edge)
        wed.left_polygon[(a, c)] = left
        wed.left_polygon[(c, b)] = left
    if rev_edge in wed.right_polygon.keys():
        right = wed.right_polygon.pop(rev_edge)
        wed.right_polygon[(b, c)] = right
        wed.right_polygon[(c, a)] = right
    if rev_edge in wed.left_polygon.keys():
        left = wed.left_polygon.pop(rev_edge)
        wed.left_polygon[(b, c)] = left
        wed.left_polygon[(c, a)] = left

    if segment:
        return wed, c
    else:
        return wed


def segment_edges(wed, distance=None, count=None):
    '''
    Segment all of the edges in the network at either
    a fixed distance or a fixed number of segments.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    distance: float distance at which edges are split
    count: integer count of the number of desired segments
    '''
    if count != None:
        assert(type(count) == int)

    def segment(count, distance, wed, start, end):
        '''
        Recursive segmentation of each edge until count is reached.
        '''
        if count == 1:
            return wed

        edge = (start, end)
        wed, start = insert_node(wed, edge, distance, segment=True)
        segment(count - 1, distance, wed, start, end)
        return wed

    #Any segmentation has float inconsistencies.  On the order of 1x10^-14
    if count == None and distance == None or count != None and distance != None:
        print '''
        Please supply either a distance at which to
        segment edges or a count of the number of
        segments to generate per edge.
        '''
        return
    lengths = edge_length(wed, half=True)
    if count != None:
        for k, l in lengths.iteritems():
            interval = l / count
            wed = segment(count, interval, wed, k[0], k[1])
    elif distance:
        for k, l in lengths.iteritems():
            if distance >= l or l / distance == 0:
                continue
            count = l / distance
            print count
            wed = segment(count, distance, wed, k[0], k[1])
    return wed


def threshold_distance(wed, cost, node, threshold, midpoint=False):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID
    threshold: float, distance to which neighbors are included
    midpoint: Boolean to indicate whether distance is computed from the start
     node or the midpoint of the edge

    Returns
    -------
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    near = []
    v0 = node
    distance = [float('inf') for x in wed.node_list]
    distance[wed.node_list.index(v0)] = 0
    pred = [None for x in wed.node_list]
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
        if distance[v] <= threshold:
            near.append(v)
        elif distance[v] > threshold:
            break
        #4. Get the neighbors to the current node
        neighbors = get_neighbor_distances(wed, v, cost)
        for v1, indiv_cost in neighbors.iteritems():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
    return near,pred


def knn_distance(wed, cost, node, n):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID
    n: integer number of nearest neighbors
    Returns:
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    near = []  #Set because we can update distance more than once
    v0 = node
    distance = [float('inf') for x in wed.node_list]
    distance[wed.node_list.index(v0)] = 0
    pred = [None for x in wed.node_list]
    a = set()
    a.add(v0)
    while len(a) > 0:
        #Get node with the lowest value from distance
        dist = float('inf')
        for node in a:
            if distance[node] < dist:
                dist = distance[node]
                v = node
                near.append(v)
        #Remove that node from the set
        a.remove(v)
        last = v
        #4. Get the neighbors to the current node
        neighbors = get_neighbor_distances(wed, v, cost)
        for v1, indiv_cost in neighbors.iteritems():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
    near = nsmallest(n + 1, distance)
    near.remove(0)  #Remove obs from near
    for i,n in enumerate(near):
        near[i] = distance.index(n)
    return near

def lat2Network(k):
    """helper function to create a network from a square lattice.

    Used for testing purposes
    """
    lat = ps.lat2W(k+1,k+1)
    k1 = k+1
    nodes = {}
    edges = []
    for node in lat.id_order:
        for neighbor in lat[node]:
            edges.append((node,neighbor))
        nodes[node] = ( node/k1, node%k1 )

    res = {"nodes": nodes, "edges": edges}

    return res


def polyShp2Network(shpFile):
    nodes = {}
    edges = {}
    f = ps.open(shpFile, 'r')
    for shp in f:
        verts = shp.vertices
        nv = len(verts)
        for v in range(nv-1):
            start = verts[v]
            end = verts[v+1]
            nodes[start] = start
            nodes[end] = end
            edges[(start,end)] = (start,end)
    f.close()
    return {"nodes": nodes, "edges": edges.values() }


def euler_nonplaner_test(e, v):
    """
    Testing for nonplanarity based on necessary condition for planarity

    Parameters
    ----------

    e: int
       number of edges
    v: int
       number of vertices


    Returns
    -------

    True if planarity condition is violated, otherwise false.

    Notes
    -----

    This is only a necessary but not sufficient condition for planarity. In
    other words violating this means the graph is nonplanar, but passing it
    does not guarantee the graph is planar.

    """

    if e <= (3*v - 6):
        return False
    else:
        return True

def area2(A, B, C):
    return (B[0]-A[0]) * (C[1]-A[1]) - (C[0]-A[0]) * (B[1]-A[1])

def isLeft(A, B, C):
    return area2(A,B,C) > 0

def isRight(A, B, C):
    return area2(A,B,C) < 0

def isCollinear(A, B, C):
    return area2(A, B, C) == 0

def intersect(A, B, C, D):

    if isLeft(A, B, D) * isRight(A, B, C):
        return True
    elif isLeft(A, B, C) * isRight(A, B, D):
        return True
    elif isCollinear(A, B, C):
        return True
    elif isCollinear(A, B, D):
        return True
    else:
        return False


def intersection_sweep(segments, findAll = True):
    """
    Plane sweep segment intersection detection.


    Parameters
    ----------

    segments: list of lists
              each segment is a list containing tuples of segment start/end points

    findAll : boolean
              If True return all segment intersections, otherwise stop after
              first detection

    Examples
    --------

    >>>segments = [ [(4.5,0), (4.5,4.5)], [(4.5,1), (4.5,2)], [(4,4), (1,4)], [(2,3), (5,3)], [(5,0), (5,10)] ]
    >>>util.intersection_seep(segments)
    [(0,3), (4,3)]
    >>>util.intersection_seep(segments, findAll=False)
    [(0,3)]

    """               
    Q = []
    slopes = []
    intercepts = []
    for i,seg in enumerate(segments):
        seg.sort()
        l,r = seg
        Q.append([l,i])
        Q.append([r,i])

        m = r[1] - l[1]  
        dx = r[1] - r[0]
        if dx == 0:
            m = 0
            intercept = r[1]
        else:
            m = m / dx
            intercept = r[1] - m * r[0]
        slopes.append(m)
        intercepts.append(intercept)

    Q.sort()  # event point que sorted on x coord
    status = []

    visited = [0] * len(segments)
    intersections = []

    while Q:
        event_point, i = Q.pop(0)
        if visited[i]:
            # right end point so we are leaving
            # check for intersection between i's left and right neighbors on
            # the status
            if position > 0 and position < ns-1:
                left = sorted_y[position-1][1] 
                right = sorted_y[position+1][1]
                p0,p1 = segments[left]
                p2,p3 = segments[right]
                if intersect(p0, p1, p2, p3):
                    intersections.append( (left,right) )
                    if not findAll:
                        Q = []
            # remove i from status
            status.remove(i)
        else:
            sorted_y = []
            visited[i] = 1
            xi = event_point[0]
            yi = event_point[1]
            sorted_y.append( (yi, i) )

            # insert in status
            for seg in status:
                y = slopes[seg] * xi + intercepts[seg]
                sorted_y.append( (y, seg) )
            sorted_y.sort()
            position = sorted_y.index( (yi, i) )
            ns = len(sorted_y)
            # check for intersection with left neighbor
            if position  > 0:
                left = sorted_y[position-1][1] 
                p0,p1 = segments[left]
                p2,p3 = segments[i]
                if intersect(p0, p1, p2, p3):
                    intersections.append( (left,i) )
                    if not findAll:
                        Q = []

            # check for intersection with right neighbor
            if position < ns-1:
                right = sorted_y[position+1][1] 
                p0,p1 = segments[right]
                p2,p3 = segments[i]
                if intersect(p0, p1, p2, p3):
                    intersections.append( (i, right))
                    if not findAll:
                        Q = []
            status.append(i)

    return intersections








def _test():
    import doctest
    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    #doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == '__main__':
    #_test()
    pass

