# Analytic functions/classes for network module

"""
 - Global Network Autocorrelation (gincs)
 - Global K-Functions
 - Local Indicators of Network-Constrained Clusters (lincs)
 - Local K-Functions
 - Network Kernels
 - Accessibility Indices
"""

import networkw
import pysal as ps

def gincs(wed, y, permutations=999, segment=False):
    if segment:
        # segment wed and y
        # get new wed and extract new y
        raise NotImplementedError

    w = networkw.w_links(wed)
    mi = ps.Moran(y, w, permutations=permutations)
    return mi

def lincs(wed, y, permutations=999, segment=False):
    if segment:
        # segment wed and y
        # get new wed and extract new y
        raise NotImplementedError

    w = networkw.w_links(wed)
    # lisa from PySAL
    lisa = ps.Moran_Local(y, w, permutations=permutations)
    return lisa

