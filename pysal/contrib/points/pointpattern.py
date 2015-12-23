"""
Point Pattern Class

author: Serge Rey <sjsrey@gmail.com>

"""
import numpy as np
import sys
from pysal.cg import KDTree
from centrography import hull
from window import as_window,  poly_from_bbox
from util import cached_property
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


if sys.version_info[0] > 2:
    xrange = range


class PointPattern(object):
    """
    PointPattern Class 2-D
    """
    def __init__(self, points, window=None, marks=None):

        """

        Arguments
        ---------
        points:  array (n x p)
        """
        self.points = np.asarray(points)
        self._n, self._p = self.points.shape
        if window is None:
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        else:
            self.set_window(window)
        self._marks = marks

    def set_window(self, window):
        try:
            self._window = window
        except:
            print("not a valid Window object")

    def get_window(self):
        if not hasattr(self, '_window') or self._window is None:
            # use bbox as window
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        return self._window

    window = property(get_window, set_window)

    def summary(self):
        print('Point Pattern')
        print("{} points".format(self.n))
        print("Bounding rectangle [({},{}), ({},{})]".format(*self.mbb))
        print("Area of window: {}".format(self.window.area))
        lam_window = self.n / self.window.area
        print("Intensity estimate for window: {}".format(lam_window))

    def plot(self, window=False, title="Point Pattern", hull=False):
        x, y = self.points.T
        fig, ax = plt.subplots()
        plt.plot(x, y, '.')
        plt.title(title)
        if window:
            patches = []
            for part in self.window.parts:
                p = Polygon(np.asarray(part))
                patches.append(p)
            ax.add_collection(PatchCollection(patches, facecolor='w',
                              edgecolor='k', alpha=0.3))
        if hull:
            patches = []
            p = Polygon(self.hull)
            patches.append(p)
            ax.add_collection(PatchCollection(patches, facecolor='w',
                              edgecolor='k', alpha=0.3))

        ax.autoscale_view()
        plt.plot(x, y, '.')

    def _mbb(self):
        """
        Minimum bounding box
        """
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return np.hstack((mins, maxs))

    mbb = cached_property(_mbb)

    def _mbb_area(self):
        """
        Area of minimum bounding box
        """

        return np.product(self.mbb[[2, 3]]-self.mbb[[0, 1]])

    mbb_area = cached_property(_mbb_area)

    def _n(self):
        """
        Number of points
        """
        return self.points.shape[0]

    n = cached_property(_n)

    def _lambda_mbb(self):
        """
        Intensity based on minimum bounding box
        """
        return self.n * 1. / self.mbb_area

    lambda_mbb = cached_property(_lambda_mbb)

    def _hull(self):
        """
        Points defining convex hull in counterclockwise order
        """
        return hull(self.points)

    hull = cached_property(_hull)

    def _hull_area(self):
        """
        Area of convex hull
        """
        h = self.hull
        if not np.alltrue(h[0] == h[-1]):
            # not in closed cartographic form
            h = np.vstack((h, h[0]))
        s = h[:-1, 0] * h[1:, 1] - h[1:, 0] * h[:-1, 1]
        return s.sum() / 2.

    hull_area = cached_property(_hull_area)

    def _lambda_hull(self):
        """
        Intensity based on convex hull
        """
        return self.n * 1. / self.hull_area

    lambda_hull = cached_property(_lambda_hull)

    def _build_tree(self):
        return KDTree(self.points)
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
        other: PointPattern
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


if __name__ == "__main__":
    # table 4.9 O'Sullivan and Unwin 2nd edition. Note observation 9 is
    # incorrect on its coordinates. Have to update (approximation for now)

    points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21], [9.47, 31.02],
              [30.78, 60.10], [75.21, 58.93], [79.26,  7.68], [8.23, 39.93],
              [98.73, 80.53], [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]

    pp = PointPattern(points)
