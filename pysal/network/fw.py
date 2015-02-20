from collections import defaultdict
import pysal as ps
import numpy as np

def floyd_warshall(ntw, cost = None, directed = False):
    """
    Uses Floyd & Warshall's algorithm in O(n^3) time using O(nm) initialization
    and O(n^2) space. 

    Parameters
    ----------
    ntw : pysal network object
    cost : dict
           key is tuple of start,end nodes of the edge
           value is the cost of traversing the arc
           if not set, defaults to the self.edge_lengths value of ntw
    directed : bool
               True if the network is directed
               False if the network is not directed
               default is False, meaning the edges are added both forwards and
               backwards, i.e. (2,3) contains edges (2,3) and (3,2). 

    Returns
    -------
    dist: dict of dicts
          outer dictinoary key is the node id at the start of the path (start)
          inner dictionary key is the node id at the end of the path (dest)
          inner dictionary value is the distance from outer key to inner key

    pred: dict of dicts
          outer dictionary key is the node id at the start of the path (start)
          inner dictionary key is the node id at the end of the path (dest)
          inner dictionary value lists steps from start ending at dest
    """
    dist = defaultdict(lambda : defaultdict(lambda : np.inf)) #defaultdict requires callable
    pred = defaultdict(dict)
    
    if not cost:
        cost = ntw.edge_lengths

    #populate initial predecessor and distance dictionaries
    for node,neighbors in ntw.adjacencylist.iteritems():
        for neighbor in neighbors:
            if (node,neighbor) in cost.keys():
                dist[node][neighbor] = cost[(node,neighbor)]
                pred[neighbor][node] = [node] #forward: node -> neighb
                if not directed:
                    pred[node][neighbor] = [neighbor] #backward: neighb -> node 
            elif (neighbor,node) in cost.keys():
                dist[node][neighbor] = cost[(neighbor, node)]
                pred[node][neighbor] = [neighbor] #forward: neighb -> node
                if not directed:
                    pred[neighbor][node] = [node] #backward: node -> neighb
        dist[node][node] = 0
        pred[node][node] = []
#    return dist, pred

    #update precedence and distance using intermediate paths
    for inter in ntw.node_list:
        for start in ntw.node_list:
            for dest in ntw.node_list:
                if dist[start][dest] > dist[start][inter] + dist[inter][dest]:
                    dist[start][dest] = dist[start][inter] + dist[inter][dest]
                    pred[start][dest] = pred[start][inter] + pred[inter][dest]
    return dist, pred
