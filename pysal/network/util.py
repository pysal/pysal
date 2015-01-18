from collections import OrderedDict
import math
import operator

import numpy as np


def nearestneighborsearch(obs_to_node, alldistances, endnode, dist):
    """
    Given a node on a network which is tagged to an observation, find the
    nearest node which also has one or more observations.
    """
    searching = True
    #sorted dict of nodes by distance
    for k, v in alldistances[endnode][0].iteritems():
        #List of the neighbors tagged to the node
        possibleneighbors = obs_to_node[k]
        if possibleneighbors:
            for n in possibleneighbors:
                if n == v:
                    continue
                else:
                    nearest_obs = n
                    nearest_node = k
                    nearest_node_distance = v + dist
                    searching = False
        if searching == False:
            break

    return nearest_obs, nearest_node, nearest_node_distance

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
    for p in pred:
        idx = p
        path = [idx]
        while idx > 0:
            nextnode = pred[idx]
            idx = nextnode
            if idx > 0:
                path.append(nextnode)
        if p > 0:
            tree[p] = path
    return tree


def cumulativedistances(distance, tree):
    distances = {}
    for k, v in tree.iteritems():
        subset_distance = distance[v]
        distances[k] = np.sum(subset_distance)
    return OrderedDict(sorted(distances.iteritems(), key=operator.itemgetter(1)))


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


def shortest_path(ntw, cost, start, end):
    distance, pred = dijkstra(ntw, cost, start)
    path = [end]
    previous = pred[end]
    while previous != start:
        path.append(previous)
        end = previous
        previous = pred[end]
    path.append(start)
    return tuple(path)

def nearest_neighbor_search(pt_indices, dist_to_node, obs_to_node, alldistances, snappedcoords):
    nearest = np.empty((2,len(pt_indices)))

    for i, p1 in enumerate(pt_indices):
        dist1, dist2 = dist_to_node[p1].values()
        endnode1, endnode2 = dist_to_node[p1].keys()

        snapped_coords = snappedcoords[p1]
        nearest_obs1, nearest_node1, nearest_node_distance1 = nearestneighborsearch(obs_to_node, alldistances, endnode1, dist1)
        nearest_obs2, nearest_node2, nearest_node_distance2 = nearestneighborsearch(obs_to_node, alldistances, endnode2, dist2)

        if nearest_node_distance2 <= nearest_node_distance1:
            nearest[i,0] = nearest_obs2
            nearest[i,1] = nearest_node_distance2
        else:
            nearest[i,0] = nearest_obs1
            nearest[i,1] = nearest_node_distance1

    return nearest
