"""
A library of spatial network k-function functions.
Not to be used without permission.

Contact: 

Andrew Winslow
GeoDa Center for Geospatial Analysis
Arizona State University
Tempe, AZ
Andrew.Winslow@asu.edu
"""

import unittest
import test 

def _fxrange(start, end, incr):
    """
    A float version of the xrange() built-in function.

    _fxrange(number, number, number) -> iterator

    Arguments:
    start -- the lower end of the range (inclusive)
    end -- the upper end of the range (exclusive)
    incr -- the step size. must be positive.
    """
    i = 0
    while True:
        t = start + i*incr
        if t >= end:
            break
        yield t
        i += 1    

def _binary_search(list, q):
    """
    Returns the index in a list where an item should be found.
     
    Arguments:
    list -- a list of items
    q -- a value to be searched for
    """
    l = 0
    r = len(list)
    while l < r:
        m = (l + r)/2
        if list[m] > q:
            r = m
        else:
            l = m + 1
    return l

def kt_values(t_specs, distances, scaling_const):
    """
    Returns a dictionary of t numerics to k(t) numerics.

    kt_values(number list, number list, number) -> number to number dictionary

    Arguments:
    t_specs -- a 3-tuple of (t_min, t_max, t_delta) specifying the t-values to compute k(t) for
    distances -- a list of distances to compute k(t) from
    scaling_const -- a constant to multiple k(t) by for each t 
    """
    ks = {}
    distances.sort()
    if type(t_specs) == tuple:
        t_specs = [t for t in _fxrange(t_specs[0], t_specs[1], t_specs[2])]
        
    for t in t_specs:
        ks[t] = scaling_const*_binary_search(distances, t)
    return ks     



