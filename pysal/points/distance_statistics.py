"""
Distance based point pattern statistics

Author: Serge Rey <sjsrey@gmail.com>


TODO

- flesh out Point_Pattern class
- doc strings
- unit tests
- simulation based inference
- analytical inference
"""


import sys
import math
import random
import numpy as np
import pysal

MAXD = sys.float_info.max
MIND = sys.float_info.min

import functools


def cached_property(fun):
    """A memoize decorator for class properties.

    Adapted from: http://code.activestate.com/recipes/576563-cached-property/
    """
    @functools.wraps(fun)
    def get(self):
        try:
            return self._cache[fun]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        ret = self._cache[fun] = fun(self)
        return ret
    return property(get)


class Point_Pattern(object):
    """
    Point Pattern

    """
    def __init__(self, points):
        """

        Arguments
        ---------
        points:  array (n x p)
        """
        self.points = np.array(points)
        self.n, self.p = self.points.shape

    def _build_tree(self):
        return pysal.cg.KDTree(self.points)

    tree = cached_property(_build_tree)

    def knn(self, k=1):
        """
        Find k nearest neighbors for each point in the pattern

        Arguments
        ---------
        k:      int
                number of nearest neighbors to find

        Returns
        -------
        nn:    array (n x k)
               row i  column j contains the id for i's jth nearest neighbor

        nnd:   array(n x k)
               row i column j contains the distance between i and its jth
               nearest neighbor
        """

        nn = self.tree.query(self.tree.data, k=k+1)
        return nn[1][:, 1:], nn[0][:, 1:]

    def _nn_sum(self):
        """
        Nearest neighbor distances
        """
        ids, nnd = self.knn(1)
        return nnd

    nnd = cached_property(_nn_sum)  # nearest neighbor distances

    def _min_nnd(self):
        """
        Min nearest neighbor distance
        """
        return self.nnd.min()

    min_nnd = cached_property(_min_nnd)

    def _max_nnd(self):
        """
        Max nearest neighbor distance
        """
        return self.nnd.max()

    max_nnd = cached_property(_max_nnd)

    def _mean_nnd(self):
        return self.nnd.mean()

    mean_nnd = cached_property(_mean_nnd)

    def G(self, intervals=10):
        """
        G function: cdf for nearest neighbor distances

        Arguments
        ---------
        intervals: int
                   number of intervals to evaluate G over

        Returns
        -------
        cdf: array (intervals x 2)
             first column is d, second column is cdf(d)
        """
        w = self.max_nnd/intervals
        n = len(self.nnd)
        d = [w*i for i in range(intervals + 2)]
        cdf = [0] * len(d)
        for i, d_i in enumerate(d):
            smaller = [nndi for nndi in self.nnd if nndi <= d_i]
            cdf[i] = len(smaller)*1./n
        return np.vstack((d, cdf)).T


    ### Pick up here

    def csr(self, n, ranges):
        """
        Generate a CSR pattern of size n in p space

        Arguments
        ---------
        n: int
           number of points to generate

        range: array (2 x p)
            column i has the min and max value of dimension i

        Returns
        -------
        y: array (n x p)
           csr realization of size n in p-space
        """
        return NotImplemented

    def F(self, n=100, intervals=10):
        """
        F: empty space function

        Arguments
        ---------
        n: int
           number of empty space points
        intevals: int
            number of intervals to evalue F over

        Returns
        -------
        cdf: array (intervals x 2)
             first column is d, second column is cdf(d)

        """
        return NotImplemented

    def J(self, n=100, intervals=10):
        """
        J: scaled G function

        Arguments
        ---------
        n: int
           number of empty space points
        intevals: int
            number of intervals to evalue F over

        Returns
        -------
        cdf: array (intervals x 2)
             first column is d, second column is cdf(d)
        """
        return NotImplemented

    def K(self, intervals=10):
        """
        Ripley's K function

        Arguments
        ---------

        """
        return NotImplemented



    def find_pairs(self, r):
        """
        Find all pairs of points in the pattern that are within r units of each
        other

        Arguments
        ---------
        r: float
           diameter of pair circle

        Returns
        ------
        s: set
           pairs of points within r units of each other

        """
        return self.tree.query_pairs(r)

    def knn_other(self, other, k=1):
        """
        Find k nearest neighbors in the pattern for each point in other

        Arguments
        ---------
        other: Point_Pattern
                n points on p dimensions

        k:      int
                number of nearest neighbors to find

        Returns
        -------
        nn:    array (n x k)
               row i  column j contains the id for i's jth nearest neighbor

        nnd:   array(n x k)
               row i column j contains the distance between i and its jth
               nearest neighbor
        """

        nn = self.tree.query(other.points, k=k+1)
        return nn[1][:, 1:], nn[0][:, 1:]






def nn_distances(points):
    tree = pysal.cg.KDTree(points)
    nn = tree.query(tree.data, k=2)
    return nn[0][:, 1]


def nn_ids(points):
    tree = pysal.cg.KDTree(points)
    nn = tree.query(tree.data, k=2)
    return nn[1][:, 1]


def nn_distances_bf(points):
    """
    Brute force nearest neighbors
    """
    n = len(points)
    d_mins = [MAXD] * n
    neighbors = [-1] * n
    for i, point_i in enumerate(points[:-1]):
        i_x, i_y = point_i
        for j in range(i+1, n):
            point_j = points[j]
            j_x, j_y = point_j
            dx = i_x - j_x
            dy = i_y - j_y
            d_ij = dx*dx + dy*dy
            if d_ij < d_mins[i]:
                d_mins[i] = d_ij
                neighbors[i] = j
            if d_ij < d_mins[j]:
                d_mins[j] = d_ij
                neighbors[j] = i
    d_mins = [math.sqrt(d_i) for d_i in d_mins]
    return neighbors, d_mins


def d_min_bf(points):
    """
    Brute force mean nearest neighbor statistic

    """
    neighbors, d_mins = nn_distances(points)
    n = len(d_mins)
    return sum(d_mins)/n


def G_bf(points, k=10):
    """
    Brute force cumulative frequency distribution of nearest neighbor
    distances
    """
    neighbors, d_mins = knn(points, k=1)

    d_max = max(d_mins)
    w = d_max/k
    n = len(d_mins)

    d = [w*i for i in range(k+2)]
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [d_i_min for d_i_min in d_mins if d_i_min <= d_i]
        cdf[i] = len(smaller)*1./n
    return d, cdf


def mbr_bf(points):
    """
    Minimum bounding rectangle, brute force
    """
    min_x = min_y = MAXD
    max_x = max_y = MIND
    for point in points:
        x, y = point
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y
    return min_x, min_y, max_x, max_y


def F_bf(points, n=100):
    x0, y0, x1, y1 = mbr_bf(points)
    ru = random.uniform
    r_points = [(ru(x0, x1), ru(y0, y1)) for i in xrange(n)]
    d_mins = [MAXD] * n
    neighbors = [-9] * n
    for i, r_point in enumerate(r_points):
        d_i = MAXD
        x0, y0 = r_point
        for j, point in enumerate(points):
            x1, y1 = point
            dx = x0-x1
            dy = y0-y1
            d = dx*dx + dy*dy
            if d < d_i:
                d_mins[i] = d
                neighbors[i] = j
                d_i = d
    return [math.sqrt(d_min_i) for d_min_i in d_mins], neighbors


def F_cdf_bf(points, n=100, k=10,):
    d, g_cdf = G_bf(points, k=k)
    d_mins, neighbors = F_bf(points, n)
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [d_i_min for d_i_min in d_mins if d_i_min <= d_i]
        cdf[i] = len(smaller)*1./n
    return d, cdf


def k_bf(points, n_bins=100):
    n = len(points)
    x0, y0, x1, y1 = mbr_bf(points)
    d_max = (x1-x0)**2 + (y1-y0)**2
    d_max = math.sqrt(d_max)
    w = d_max / (n_bins-1)
    d = [w*i for i in range(n_bins)]
    ks = [0] * len(d)
    for i, p_i in enumerate(points[:-1]):
        x0, y0 = p_i
        for j in xrange(i+1, n):
            x1, y1 = points[j]
            dx = x1-x0
            dy = y1-y0
            dij = math.sqrt(dx*dx + dy*dy)
            uppers = [di for di in d if di >= dij]
            for upper in uppers:
                ki = d.index(upper)
                ks[ki] += 2
    return ks, d


def csr_bf(bb, n=100, n_conditioned=True):

    x0, y0, x1, y1 = bb
    ru = random.uniform
    if n_conditioned:
        points = [(ru(x0, x1), ru(y0, y1)) for i in xrange(n)]
    else:
        ns = np.random.poisson(n)
        points = [(ru(x0, x1), ru(y0, y1)) for i in xrange(ns)]
    return points


if __name__ == '__main__':

    # table 4.9 O'Sullivan and Unwin 2nd edition. Note observation 9 is
    # incorrect on its coordinates. Have to update (approximation for now)

    points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21], [9.47, 31.02],
              [30.78, 60.10], [75.21, 58.93], [79.26,  7.68], [8.23, 39.93],
              [98.73, 80.53], [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]

    p1 = np.array(points)
    p2 = p1[:5:,:]  + 100

    p1 = Point_Pattern(p1)
    p2 = Point_Pattern(p2)
