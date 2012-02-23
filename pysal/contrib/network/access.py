"""
A library of spatial network accessibility functions.
Not to be used without permission.

Contact: 

Andrew Winslow
GeoDa Center for Geospatial Analysis
Arizona State University
Tempe, AZ
Andrew.Winslow@asu.edu
"""

import math
import unittest

import test


def coverage(dists, bandwidth):
    """
    Takes a list of numeric distances and a numeric bandwidth and returns the 
    number of distances less than or equal to the bandwidth.
    """
    return len(filter(lambda d: d <= bandwidth, dists))

def equity(dists):
    """
    Takes a list of numeric distances and returns the smallest of them.
    """
    return min(dists)

def potential_entropy(dists, power=1):
    """
    Takes a list of numeric distances and returns the sum of the values
    of a function of a distances. The function is e^(-power*distance).  
    """
    return sum([math.e**(-power*d) for d in dists])

def potential_gravity(dists, power=2):
    """
    Takes a list of numeric distances and returns the sum of the values
    of a function of a distances. The function is 1/(d^power).
    """
    return sum([1.0/(d**power) for d in filter(lambda d: d > 0, dists)])

def travel_cost(dists):
    """
    Takes a list of distances and compute the sum. 
    """
    return sum(dists)
   
    



