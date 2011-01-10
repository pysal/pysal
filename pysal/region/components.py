"""
Checking for connected components in a graph.
"""        
__author__  = "Sergio J. Rey <srey@asu.edu> "

import sys
from operator import gt, lt

__all__ = ["check_contiguity"]
        
def check_contiguity(w,neighbors,leaver):
    """Check if contiguity is maintained if leaver is removed from neighbors


    Parameters
    ==========

    w: spatial weights object
        simple contiguity based weights

    neighbors: list
        nodes that are to be checked if they form a single connected component

    leaver: id
        a member of neighbors to check for removal


    Returns
    =======

    True: if removing id from leaver does not break contiguity of remaing set
        in neighbors

    False: if removing id from neighbors breaks contiguity

    Example
    =======

    >>> from pysal.weights import lat2gal
    >>> w=lat2gal()
    >>> check_contiguity(w,[0,1,2,3,4],4)
    True
    >>> check_contiguity(w,[0,1,2,3,4],3)
    False
    >>> check_contiguity(w,[0,1,2,3,4],0)
    True
    >>> check_contiguity(w,[0,1,2,3,4],1)
    False
    >>> 
    """
    d={}
    g=Graph()
    for i in neighbors:
        d[i]=[j for j in w.neighbors[i] if (j in neighbors and j != leaver)]
    try:
        d.pop(leaver)
    except:
        pass
    for i in d:
        for j in d[i]:
            g.add_edge(i,j,1.0)
    cc=g.connected_components(op=gt)
    if len(cc)==1:
        neighbors.remove(leaver)
        if cc[0].nodes == set(neighbors):
            return True 
        else:
            return False
    else:
        return False

class Graph(object):
    def __init__(self):
        self.nodes=set()
        self.edges={}
        self.cluster_lookup={}
        self.no_link={}

    def add_edge(self,n1,n2,w):
        self.nodes.add(n1)
        self.nodes.add(n2)
        self.edges.setdefault(n1,{}).update({n2:w})
        self.edges.setdefault(n2,{}).update({n1:w})

    def connected_components(self,threshold=0.9, op=lt):
        nodes = set(self.nodes)
        components,visited =[], set()
        while len(nodes) > 0:
            connected, visited = self.dfs(nodes.pop(), visited, threshold, op)
            connected = set(connected)
            for node in connected:
                if node in nodes:
                    nodes.remove(node)
            subgraph=Graph()
            subgraph.nodes = connected
            subgraph.no_link = self.no_link
            for s in subgraph.nodes:
                for k,v in self.edges.get(s,{}).iteritems():
                    if k in subgraph.nodes:
                        subgraph.edges.setdefault(s,{}).update({k:v})
                if s in self.cluster_lookup:
                    subgraph.cluster_lookup[s] = self.cluster_lookup[s]
            components.append(subgraph)
        return components
    
    def dfs(self, v, visited, threshold, op=lt, first=None):
        aux=[v]
        visited.add(v)
        if first is None:
            first = v
        for i in (n for n, w in self.edges.get(v,{}).iteritems() \
                  if op(w, threshold) and n not in visited):
            x,y=self.dfs(i,visited,threshold,op,first)
            aux.extend(x)
            visited=visited.union(y)
        return aux, visited

# tests

def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
