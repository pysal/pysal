"""
A library of spatial network functions.
Not to be used without permission.

Contact: 

Andrew Winslow
GeoDa Center for Geospatial Analysis
Arizona State University
Tempe, AZ
Andrew.Winslow@asu.edu
"""

import csv
import numpy as np
from pysal import W
import unittest
import test

def dist_weights(distfile, weight_type, ids, cutoff, inverse=False):
    """
    Returns a distance-based weights object using user-defined options
    
    Parameters
    ----------
    distfile: string, a path to distance csv file
    weighttype: string, either 'threshold' or 'knn'
    ids: a numpy array of id values
    cutoff: float or integer; float for 'threshold' weight type and integer for knn type
    inverse: boolean; true if inversed weights required

    """
    try:
        data_csv = csv.reader(open(distfile))        
        if csv.Sniffer().has_header(distfile):
            data_csv.next()
    except:        
        data_csv = None
    
    if weight_type == 'threshold':
        def neighbor_func(dists, threshold):
            dists = filter(lambda x: x[0] <= threshold, dists)
            return dists
    else:
        def neighbor_func(dists, k):
            dists.sort()
            return dists[:k]

    if inverse:
        def weight_func(dists, alpha=-1.0):
            return list((np.array(dists)**alpha).round(decimals=6))
    else:
        def weight_func(dists, binary=False):
            return [1]*len(dists)

    dist_src = {}
    for row in data_csv:
        des = dist_src.setdefault(row[0], {})
        if row[0] != row[1]:
            des[row[1]] = float(row[2])

    neighbors, weights = {}, {}
    for id_val in ids:
        if id_val not in dist_src:
            raise ValueError, 'An ID value doest not exist in distance file'
        else:
            dists = zip(dist_src[id_val].values(), dist_src[id_val].keys())
        ngh, wgt = [], []
        if len(dists) > 0:
            nghs = neighbor_func(dists, cutoff)
            for d, i in nghs:
                ngh.append(i)
                wgt.append(d)
        neighbors[id_val] = ngh
        weights[id_val] = weight_func(wgt)
    w = W(neighbors, weights)
    w.id_order = ids
    return w


