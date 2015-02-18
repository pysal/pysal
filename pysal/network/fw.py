from collections import defaultdict
import pysal as ps
import numpy as np

ntw = ps.network.network.Network(ps.examples.get_path('geodanet/streets.shp'))

def floyd_warshall(ntw, cost = None, directed = False):
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
