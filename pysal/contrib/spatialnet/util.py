"""
Utility module for network contrib 
"""

import pysal as ps
import networkx as nx
import numpy as np

__author__ = "Serge Rey <sjsrey@gmail.com>"

def w2dg(w):
    """
    Return a networkx directed graph from a PySAL W object


    Parameters
    ----------

    w: Weights 


    Returns
    -------
    G: A networkx directed graph


    Example
    ------

    >>> import networkx as nx
    >>> import pysal as ps
    >>> w = ps.lat2W()
    >>> guw = w2dg(w)
    >>> guw.in_degree()
    {0: 2, 1: 3, 2: 3, 3: 3, 4: 2, 5: 3, 6: 4, 7: 4, 8: 4, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4, 14: 3, 15: 3, 16: 4, 17: 4, 18: 4, 19: 3, 20: 2, 21: 3, 22: 3, 23: 3, 24: 2}
    >>> dict([(k,len(w.neighbors[k])) for k in w.neighbors])
    {0: 2, 1: 3, 2: 3, 3: 3, 4: 2, 5: 3, 6: 4, 7: 4, 8: 4, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4, 14: 3, 15: 3, 16: 4, 17: 4, 18: 4, 19: 3, 20: 2, 21: 3, 22: 3, 23: 3, 24: 2}
    >>> 
    """

    w_l = [(i,j) for i in w.neighbors for j in w[i]]
    G = nx.DiGraph()
    G.add_edges_from(w_l)
    return G

def w2dwg(w):
    """
    Return a directed, weighted graph from a PySAL W object


    Parameters
    ----------

    w: Weights 


    Returns
    -------
    G: A networkx directed, weighted graph


    Example
    -------
    >>> import networkx as nx
    >>> import pysal as ps
    >>> w = ps.lat2W()
    >>> w.transform = 'r'
    >>> gw = w2dwg(w)
    >>> gw.get_edge_data(0,1)
    {'weight': 0.5}
    >>> gw.get_edge_data(1,0)
    {'weight': 0.33333333333333331}
    """

    w_l = [(i,j,w[i][j]) for i in w.neighbors for j in w[i]]
    G = nx.DiGraph() # allow for asymmetries in weights
    G.add_weighted_edges_from(w_l)
    return G


def dwg2w(g, weight_name = 'weight'):
    """
    Returns a PySAL W object from a directed-weighted graph

    Parameters
    ----------

    g: networkx digraph

    weight_name: name of weight attribute of g

    Returns
    -------
    w: PySAL W 

    Example
    -------
    >>> w = ps.lat2W()
    >>> w.transform = 'r'
    >>> g = w2dwg(w)
    >>> w1 = dwg2w(g)
    >>> w1.n
    25
    >>> w1.neighbors[0]
    [1, 5]
    >>> w1.neighbors[1]
    [0, 2, 6]
    >>> w1.weights[0]
    [0.5, 0.5]
    >>> w1.weights[1]
    [0.33333333333333331, 0.33333333333333331, 0.33333333333333331]
    """

    neighbors = {}
    weights = {}
    for node in g.nodes_iter():
        neighbors[node] = []
        weights[node] = []
        for neighbor in g.neighbors_iter(node):
            neighbors[node].append(neighbor)
            weight = g.get_edge_data(node,neighbor)
            if weight:
                weights[node].append(weight[weight_name])
            else:
                weights[node].append(1)
    return ps.W(neighbors=neighbors, weights=weights)




def edge2w(edgelist, nodetype=str):
    """
    Create a PySAL W object from an edgelist

    Parameters
    ----------

    edge_file: file with edgelist

    nodetype: type for node (str, int, float)


    Returns
    -------
    W: PySAL W


    Example
    -------
    >>> lines = ["1 2", "2 3", "3 4", "4 5"]
    >>> w = edge2w(lines)
    >>> w.n
    5
    >>> w.neighbors["2"]
    ['1', '3']

    >>> w = edge2w(lines, nodetype=int)
    >>> w.neighbors[2]
    [1, 3]
    >>> lines = ["1 2 {'weight':1.0}", "2 3 {'weight':0.5}", "3 4 {'weight':3.0}"] 
    >>> w = edge2w(lines, nodetype=int)
    >>> w.neighbors[2]
    [1, 3]
    >>> w.weights[2]
    [1.0, 0.5]

    """
    G = nx.parse_edgelist(edgelist, nodetype=nodetype)
    return dwg2w(G)

    

def adjl2w(adjacency_list, nodetype=str):
    """
    Create a PySAL W object from an adjacency list file

    Parameters
    ----------

    adjacency_list: list of adjacencies
                    for directed graphs list only outgoing adjacencies

    nodetype: type for node (str, int, float)


    Returns
    -------
    W: PySAL W


    Example
    -------
    >>> al = [[1], [0,2], [1,3], [2]]
    >>> w = adjl2w(al)
    >>> w.n
    4
    >>> w.neighbors['0']
    ['1']
    >>> w = adjl2w(al, nodetype=int)
    >>> w.n
    4
    >>> w.neighbors[0]
    [1]


    """

    adjacency_list = [ map(nodetype, neighs) for neighs in adjacency_list]
    return ps.W(dict([(nodetype(i),neighs) for i,neighs in enumerate(adjacency_list)]))

                       

if __name__ == '__main__':
    import doctest
    doctest.testmod()






