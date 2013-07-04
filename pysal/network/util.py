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

    vertices = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1,
        1), 6: (1, 2), 7: (1, 3), 8: (2, 0), 9: (2, 1    ), 10: (2, 2), 11:
        (2, 3), 12: (3, 0), 13: (3, 1), 14: (3, 2), 15: (3, 3)}
    edges = [(0, 1), (0, 4), (1, 0), (1, 2), (1, 5), (2, 1), (2, 3), (2, 6),
            (3, 2), (3, 7), (4, 0), (4, 8), (4, 5), (5, 1)    , (5, 4), (5,
                6), (5, 9), (6, 2), (6, 10), (6, 5), (6, 7), (7, 11), (7, 3),
            (7, 6), (8, 12), (8, 4), (8, 9), (9, 8), (9, 10), (9, 5    ), (9,
                13), (10, 9), (10, 11), (10, 14), (10, 6), (11, 10), (11, 15),
            (11, 7), (12, 8), (12, 13), (13, 9), (13, 12), (13, 14), (14,
                10), (14, 13), (14, 15), (15, 11), (15, 14)]
    we1 = WED(vertices,edges)

