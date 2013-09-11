"""
Utilities for PySAL Network Module
"""

__author__ = "Sergio Rey <sjsrey@gmail.com>, Jay Laura <jlaura@asu.edu>"

import copy
import math
from collections import OrderedDict
import pysal as ps
import numpy as np
from pysal.cg.standalone import get_points_dist

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

    """

    obs_to_edge = {}

    polys = {}
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
            if ps.cg.standalone.get_polygon_point_intersect(poly, point):
                d = {}

                for i in range(len(region_edges)-1):
                    region_edges[i]
                    xi = wed.node_coords[region_edges[i][0]][0]
                    xi1 = wed.node_coords[region_edges[i+1][0]][0]
                    yi = wed.node_coords[region_edges[i][1]][1]
                    yi1 = wed.node_coords[region_edges[i+1][1]][1]
                    distance = abs((xi-xi1)*(point[1] - yi) - (yi-yi1)*(point[0]-xi))\
                            / math.sqrt((xi-xi1)**2 + (yi-yi1)**2)
                    d[distance] = (xi, xi1, yi, yi1, region_edges[i])

                min_dist = OrderedDict(sorted(d.items()))

                for value in min_dist.iteritems():
                    xi = value[1][0]
                    xi1 = value[1][1]
                    yi = value[1][2]
                    yi1 = value[1][3]
                    if point[0] > xi and point[0] < xi1 or point[0] < xi and point[0] >xi1:
                        if point[1] > yi and point[1] < yi1 or point[1] < yi and point[1] >yi1:
                            x = point[0] - value[0] * (xi-xi1)
                            y = point[1] - value[0] * (yi-yi1)
                            #print pt_index,value[1][4], (x,y)
                            obs_to_edge[(value[1][4])] = {pt_index:(x,y)}
                            break

    return obs_to_edge

def dijkstra(wed, cost, node):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID

    Returns:
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    def get_neighbor_distances(wed, v0, l):
        edges = wed.enum_links_node(v0)
        neighbors = {}
        for e in edges:
            if e[0] != v0:
                neighbors[e[0]] = l[e]
            else:
                neighbors[e[1]] = l[e]
        return neighbors

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


def threshold_distance(wed, cost, node, threshold):
    """
    Compute the shortest path between a start node and
        all other nodes in the wed.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    cost: Cost per edge to travel, e.g. distance
    node: Start node ID
    threshold: float, distance to which neighbors are included

    Returns:
    distance: List of distances from node to all other nodes
    pred : List of preceeding nodes for traversal route
    """

    def get_neighbor_distances(wed, v0, l):
        edges = wed.enum_links_node(v0)
        neighbors = {}
        for e in edges:
            if e[0] != v0:
                neighbors[e[0]] = l[e]
            else:
                neighbors[e[1]] = l[e]
        return neighbors

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
        if distance[v] <= threshold:
            near.append(v)
        #4. Get the neighbors to the current node
        neighbors = get_neighbor_distances(wed, v, cost)
        for v1, indiv_cost in neighbors.iteritems():
            if distance[v1] > distance[v] + indiv_cost:
                distance[v1] = distance[v] + indiv_cost
                pred[v1] = v
                a.add(v1)
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

