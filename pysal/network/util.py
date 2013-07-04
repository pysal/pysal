"""
Utilities for PySAL Network Module
"""

__author__ = "Sergio J. Rey <srey@asu.edu>"

import pysal as ps
import numpy as np


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
    i = 0
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

