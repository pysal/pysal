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


def edge_length(wed):
    """
    Compute the cartesian length of all edges.  This is a helper
        function to allow for ratio data with spatial autocorrelation
        analysis.

    Parameters
    ----------
    None

    Returns
    -------
    length : dict {tuple(edge): float(length)}
        The length of each edge.
    """

    lengths = {}
    for edge in wed.edge_list:
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


def insert_node(wed, edge, distance):
    """
    Insert a node into an existing edge at a fixed distance from
     the start node.

    Parameters
    ----------
    wed: PySAL Winged Edge Data Structure
    edge: The edge to insert a node into
    distance: float, distance from the start node of the edge

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
    #Update the edge list
    a = edge[0]
    b = edge[1]
    c = newcoord_id
    idx = wed.edge_list.index(edge)
    wed.edge_list.pop(idx)
    wed.edge_list += [(a, c), (c, a)]
    idx = wed.edge_list.index((b, a))
    wed.edge_list.pop(idx)
    wed.edge_list += [(b, c),(c, b)]
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
    wed.start_c[(c, b)] = (a, c)
    wed.start_c[(c, b)] = (a, c)
    wed.end_c[(b, c)] = (c, a)
    wed.end_cc[(b, c)] = (c, a)
    #Update the pointer to the nodes incident to start / end of the original link
    for k, v in wed.start_c.iteritems():
        if v == edge:
            wed.start_c[k] = (v[0], c)
        elif v == rev_edge:
            wed.start_c[k] = (v[1], c)
    for k, v in wed.start_cc.iteritems():
        if v == edge:
            wed.start_cc[k] = (v[0], c)
        elif v == rev_edge:
            wed.start_cc[k] = (v[1], c)
    for k, v in wed.end_c.iteritems():
        if v == edge:
            wed.end_c[k] = (v[1], c)
        elif v == rev_edge:
            wed.end_c[k] = (v[0], c)
    for k, v in wed.end_cc.iteritems():
        if v == edge:
            wed.end_cc[k] = (v[1], c)
        elif v == rev_edge:
            wed.end_cc[k] = (v[0], c)
    #Update the node_edge pointer
    wed.node_edge[c] = (a, c)
    wed.node_edge[a] = (a, c)
    wed.node_edge[b] = (b, c)
    #update right and left polygon regions
    right = wed.right_polygon.pop(edge)
    left = wed.left_polygon.pop(edge)
    wed.right_polygon[(a, c)] = right
    wed.left_polygon[(a, c)] = left
    wed.right_polygon[(c, b)] = right
    wed.left_polygon[(c, b)] = left
    right = wed.right_polygon.pop(rev_edge)
    left = wed.left_polygon.pop(rev_edge)
    wed.right_polygon[(b, c)] = right
    wed.left_polygon[(b, c)] = left
    wed.right_polygon[(c, a)] = right
    wed.left_polygon[(c, a)] = left

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

