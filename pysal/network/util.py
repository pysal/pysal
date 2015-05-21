from collections import OrderedDict
import math
import operator

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


