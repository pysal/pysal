"""
Weights for PySAL Network Module
"""

__author__ = "Sergio Rey <sjsrey@gmail.com>, Jay Laura <jlaura@asu.edu>"

from itertools import combinations
import pysal as ps

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


