"""
Utility module for network contrib 
"""

import pysal as ps
import networkx as nx
import numpy as np

__author__ = "Serge Rey <sjsrey@gmail.com>"

def w2dg(w):
    """
    Return a directed graph from a PySAL W object


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
    {'weight': 0.3333333333333333}
    """

    w_l = [(i,j,w[i][j]) for i in w.neighbors for j in w[i]]
    G = nx.DiGraph() # allow for asymmetries in weights
    G.add_weighted_edges_from(w_l)
    return G




def edge2W(edge_file):
    """
    Create a PySAL W object from an edgelist

    Parameters
    ----------

    edge_file: file with edgelist


    Returns
    -------
    W: PySAL W


    Example
    -------
    >>> import networkx as nx
    >>> w = ps.lat2W()
    >>> w.transform = 'r'
    >>> g = w2dwg(w)
    >>> fh = open("test.edgelist",'wb')
    >>> nx.write_edgelist(g,fh)
    """
    info = np.loadtxt(edge_file)

    nodes = np.unique(info[:,[0,1]])
    neighbors = {}
    weights = {}
    for node in nodes:
        neighbors[node] = []
        weights[node] = []
    for row in info:
        neighbors[row[0]].append(row[1])
        weights[row[0]].append(row[2])

    w = ps.W(neighbors=neighbors, weights = weights)


    return w


                       

if __name__ == '__main__':
    import doctest
    doctest.testmod()






