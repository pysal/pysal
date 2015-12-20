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
import random
import numpy as np
import functools
import pysal

if sys.version_info[0] > 2:
    xrange = range


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

    def _mbb(self):
        """
        Minumum bounding box
        """
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return np.vstack((mins, maxs)).T

    mbb = cached_property(_mbb)

    def _mbb_area(self):
        """
        Area of minimum bounding box
        """
        return np.product(self.mbb[:, -1]-self.mbb[:, 0])

    mbb_area = cached_property(_mbb_area)

    def _lambda_bb(self):
        """
        Intensity based on minimum bounding box
        """
        return self.n * 1. / self.mbb_area

    lambda_bb = cached_property(_lambda_bb)

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

    def G(self, intervals=10, dmin=0.0, dmax=None):
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

    def csr(self, n):
        """
        Generate a CSR pattern of size n in the minimum bounding box for the
        pattern

        Arguments
        ---------
        n: int
           number of points to generate

        Returns
        -------
        y: array (n x p)
           csr realization of size n in p-space minium bounding box
        """

        return np.hstack([np.random.uniform(d[0], d[1], (n, 1)) for d in
                         self.mbb])

    def F(self, n=100, intervals=10, dmin=0.0, dmax=None, window='mbb'):
        """
        F: empty space function

        Arguments
        ---------
        n: int
           number of empty space points
        intevals: int
            number of intervals to evalue F over
        dmin: float
               lower limit of distance range
        dmax: float
               upper limit of distance range
               if dmax is None dmax will be set to maxnnd

        Returns
        -------
        cdf: array (intervals x 2)
             first column is d, second column is cdf(d)

        """

        if window.lower() == 'mbb':
            p = Point_Pattern(self.csr(n))
            nnids, nnds = self.knn_other(p, k=1)
            if dmax is None:
                max_nnds = self.max_nnd
            else:
                max_nnds = dmax
            w = max_nnds / intervals
            d = [w*i for i in range(intervals + 2)]
            cdf = [0] * len(d)
            for i, d_i in enumerate(d):
                smaller = [nndi for nndi in nnds if nndi <= d_i]
                cdf[i] = len(smaller)*1./n
            return np.vstack((d, cdf)).T
        else:
            return NotImplemented

    def J(self, n=100, intervals=10, dmin=0.0, dmax=None):
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
        F = self.F(n, intervals)
        G = self.G(intervals)
        FC = 1 - F[:, 1]
        GC = 1 - G[:, 1]
        last_id = len(GC) + 1
        if np.any(FC == 0):
            last_id = np.where(FC == 0)[0][0]

        return np.vstack((F[:last_id, 0], FC[:last_id]/GC[:last_id]))

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

        try:
            nn = self.tree.query(other.points, k=k)
        except:
            nn = self.tree.query(other, k=k)
        return nn[1], nn[0]


def nn_distances(points):
    tree = pysal.cg.KDTree(points)
    nn = tree.query(tree.data, k=2)
    return nn[0][:, 1]


def nn_ids(points):
    tree = pysal.cg.KDTree(points)
    nn = tree.query(tree.data, k=2)
    return nn[1][:, 1]


def csr(bb, n=100, n_conditioned=True):

    x0, y0, x1, y1 = bb
    ru = random.uniform
    if n_conditioned:
        n = np.random.poisson(n)
    points = [(ru(x0, x1), ru(y0, y1)) for i in xrange(n)]

    return points

if __name__ == '__main__':

    # table 4.9 O'Sullivan and Unwin 2nd edition. Note observation 9 is
    # incorrect on its coordinates. Have to update (approximation for now)

    points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21], [9.47, 31.02],
              [30.78, 60.10], [75.21, 58.93], [79.26,  7.68], [8.23, 39.93],
              [98.73, 80.53], [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]

    p1 = np.array(points)
    p2 = p1[:5:, :] + 100

    p1 = Point_Pattern(p1)
    p2 = Point_Pattern(p2)
