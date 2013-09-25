"""
Weights for PySAL Network Module
"""

__author__ = "Sergio Rey <sjsrey@gmail.com>, Jay Laura <jlaura@asu.edu>"

from itertools import combinations
import numpy as np
import pysal as ps
from util import threshold_distance, edge_length, knn_distance


def w_links(wed):
    """
    Generate Weights object for links in a WED

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure

    Returns

    ps.W(neighbors): PySAL Weights Dict
    """
    nodes = wed.node_edge.keys()
    neighbors = {}
    for node in nodes:
        lnks = wed.enum_links_node(node)
        # put i,j s.t. i < j
        lnks = [tuple(sorted(lnk)) for lnk in lnks]
        for comb in combinations(range(len(lnks)), 2):
            l, r = comb
            if lnks[l] not in neighbors:
                neighbors[lnks[l]] = []
            neighbors[lnks[l]].append(lnks[r])
            if lnks[r] not in neighbors:
                neighbors[lnks[r]] = []
            neighbors[lnks[r]].append(lnks[l])
    return ps.W(neighbors)


def w_distance(wed, threshold, cost=None, alpha=-1.0, binary=True, ids=None):
    '''
    Generate a Weights object based on a threshold
     distance using a WED

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    distance: float network threshold distance for neighbor membership
    cost: defaults to length, can be any cost dicationary {(edge): cost}

    Returns
    -------
    ps.W(neighbors): PySAL Weights Dict
    '''

    if cost is None:
        cost = edge_length(wed)
    if ids:
        ids = np.array(ids)
    else:
        ids = np.arange(len(wed.node_list))
    neighbors = {}
    if binary is True:
        for node in wed.node_list:
            near, pred = threshold_distance(wed, cost, node, threshold)
            neighbors[ids[node]] = near
        return ps.W(neighbors, None, ids)
    elif binary is False:
        weights = {}
        for node in wed.node_list:
            wt = []
            near, pred = threshold_distance(wed, cost, node, threshold)
            near.remove(node)
            neighbors[ids[node]] = near
            for end in near:
                path = [end]
                previous = pred[end]
                while previous != node:
                    path.append(previous)
                    end = previous
                    previous = pred[end]
                path.append(node)
                cum_cost = 0
                for p in range(len(path) - 1):
                    cum_cost += cost[(path[p], path[p + 1])]
                wt.append(cum_cost ** alpha)
            weights[ids[node]] = wt
        return ps.W(neighbors, weights, ids)


def w_knn(wed, n, cost=None, ids=None):
    '''
    Generate w Weights object based on the k-nearest
     network neighbors.

    Parameters
    ----------
    wed: PySAL Winged Edged Data Structure
    n: integer number of neighbors for each node
    cost: defaults to length, can be any cost dictionary

    Returns
    -------
    ps.W(neighbors): PySAL Weights Dict
    '''

    if cost is None:
        cost = edge_length(wed)
    if ids:
        ids = np.array(ids)
    else:
        ids = np.arange(len(wed.node_list))
    neighbors = {}
    for node in wed.node_list:
        neighbors[node] = knn_distance(wed, cost, node, n=n)
    return ps.W(neighbors, id_order=ids)
