"""
Planar Point Pattern Class

"""
import numpy as np
import sys
from pysal.lib.cg import KDTree
from .centrography import hull
from .window import as_window,  poly_from_bbox
from .util import cached_property
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

__author__ = "Serge Rey sjsrey@gmail.com"
__all__ = ['PointPattern']

if sys.version_info[0] > 2:
    xrange = range


class PointPattern(object):
    """
    Planar Point Pattern Class 2-D.

    Parameters
    ----------
    points:       array
                  (n,p), n points with p >= 2 attributes on each
                  point. Two attributes must comprise the spatial
                  coordinate pair. Default is that the first two
                  attributes are the x and y spatial coordinates.
    window:       :class:`.Window`
                  Bounding geometric object for the point pattern.
                  If not specified, window will be set to the minimum
                  bounding rectangle of the point pattern.
    names:        list
                  The names of the attributes.
    coord_names:  list
                  The names of the attributes defining the two spatial
                  coordinates.

    Examples
    --------
    >>> from pointpats import PointPattern
    >>> points = [[66.22, 32.54], [22.52, 22.39], [31.01, 81.21],
    ...           [9.47, 31.02], [30.78, 60.10], [75.21, 58.93],
    ...           [79.26,  7.68], [8.23, 39.93], [98.73, 77.17],
    ...           [89.78, 42.53], [65.19, 92.08], [54.46, 8.48]]
    >>> pp = PointPattern(points)
    >>> pp.n
    12
    >>> pp.mean_nnd
    21.612139802089246
    >>> pp.lambda_mbb
    0.0015710507711240867
    >>> pp.lambda_hull
    0.0022667153468973137
    >>> pp.hull_area
    5294.00395
    >>> pp.mbb_area
    7638.200000000001

    """
    def __init__(self, points, window=None, names=None, coord_names=None):

        # first two series in df are x, y unless coor_names and names are
        # specified

        self.df = pd.DataFrame(points)
        n, p = self.df.shape
        self._n_marks = p - 2
        if coord_names is None:
            if names is not None:
                coord_names = names[:2]
            else:
                coord_names = ['x', 'y']
        if names is None:
            col_names = coord_names
            if p > 2:
                for m in range(2, p):
                    col_names.append("mark_{}".format(m-2))
            coord_names = coord_names[:2]
        else:
            col_names = names

        self.coord_names = coord_names
        self._x, self._y = coord_names
        self.df.columns = col_names
        self.points = self.df.loc[:, [self._x, self._y]]
        self._n, self._p = self.points.shape
        if window is None:
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        else:
            self.set_window(window)

        self._facade()

    def __len__(self):
        """Return the number of points. Use the expression 'len(pp)'.

        Returns
        -------
        length : int
            The number of points in the point pattern.

        Examples
        --------
        >>> from pointpats import PointPattern
        >>> points = [[1, 3], [4, 5], [0,0]]
        >>> pp = PointPattern(points)
        >>> len(pp)
        3

        """
        return len(self.df)

    def __contains__(self, n):
        """Return True if n is a point (a tuple of coordinates), False otherwise.
        Use the expression 'n in pp'.

        Examples
        --------
        >>> from pointpats import PointPattern
        >>> points = [[1, 3], [4, 5], [0,0]]
        >>> pp = PointPattern(points)
        >>> [1, 3] in pp
        True
        """
        name = self.df.columns.values.tolist()
        return ((self.df[name[0]] == n[0]) & (self.df[name[1]] == n[1])).any()

    def set_window(self, window):
        try:
            self._window = window
        except:
            print("not a valid Window object")

    def get_window(self):
        """
        Bounding geometry for the point pattern

        :class:`.window.Window`
        """
        if not hasattr(self, '_window') or self._window is None:
            # use bbox as window
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        return self._window

    window = property(get_window, set_window)

    def summary(self):
        '''
        Description of the point pattern.
        '''

        print('Point Pattern')
        print("{} points".format(self.n))
        print("Bounding rectangle [({},{}), ({},{})]".format(*self.mbb))
        print("Area of window: {}".format(self.window.area))
        print("Intensity estimate for window: {}".format(self.lambda_window))
        print(self.head())

    def add_marks(self, marks, mark_names=None):
        if mark_names is None:
            nm = range(len(marks))
            mark_names = ["mark_{}".format(self._n_marks+1+j) for j in nm]
        for name, mark in zip(mark_names, marks):
            self.df[name] = mark
            self._n_marks += 1

    def plot(self, window=False, title="Point Pattern", hull=False,
             get_ax=False):
        """
        Plot function for a point pattern.

        Parameters
        ----------
        window  : boolean
                  If window is True, plot window of the point
                  pattern. If not, don't plot window.
        title   : string
                  Name of the figure.
        hull    : boolean
                  If hull is True, plot convex hull of the point
                  pattern. If not, don't plot convex hull.
        get_ax  : boolean
                  If get_ax is True, return the current plot ax.
                  
        Returns
        -------
        ax      : matplotlib.axes._subplots.AxesSubplot
                  Current plot ax. Only return it when get_ax is True.

        """
        fig, ax = plt.subplots()
        plt.plot(self.df[self._x], self.df[self._y], '.')
        # plt.scatter(self.df[self._x], self.df[self._y])
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

        # plt.plot(x, y, '.')
        if get_ax:
            return ax

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

    def _lambda_window(self):
        """
        Intensity estimate based on area of window

        The intensity of a point process at point :math:`s_j` can be defined
        as:

        .. math::

            \\lambda(s_j) = \\lim \\limits_{|\\mathbf{A}s_j|
            \\to 0} \\left \\{ \\frac{E(Y(\mathbf{A}s_j)}{|\mathbf{A}s_j|}
            \\right \\}

        where :math:`\\mathbf{A}s_j` is a small region surrounding location
        :math:`s_j` with area :math:`|\\mathbf{A}s_j|`, and
        :math:`E(Y(\\mathbf{A}s_j))` is the expected number of event points in
        :math:`\\mathbf{A}s_j`.

        The intensity is the mean number of event points per unit of area at
        point :math:`s_j`.

        """
        return self.n / self.window.area

    lambda_window = cached_property(_lambda_window)

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

        Parameters
        ----------
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
        if k < 1:
            raise ValueError('k must be at least 1')
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
        """
        Mean nearest neighbor distance
        """
        return self.nnd.mean()

    mean_nnd = cached_property(_mean_nnd)

    def find_pairs(self, r):
        """
        Find all pairs of points in the pattern that are within r units of each
        other

        Parameters
        ----------
        r: float
           diameter of pair circle

        Returns
        -------
        s: set
           pairs of points within r units of each other

        """
        return self.tree.query_pairs(r)

    def knn_other(self, other, k=1):
        """
        Find k nearest neighbors in the pattern for each point in other

        Parameters
        ----------
        other: :class:`PointPattern`

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
        if k < 1:
            raise ValueError('k must be at least 1')
        try:
            nn = self.tree.query(other.points, k=k)
        except:
            nn = self.tree.query(other, k=k)
        return nn[1], nn[0]

    def explode(self, mark):
        """
        Explode a marked point pattern into a sequence of individual point
        patterns. If the mark has k unique values, then the sequence will be of
        length k.

        Parameters
        ----------
        mark: string
              The label of the mark to use for the subsetting

        Returns
        -------
        pps: list
             sequence of :class:`PointPattern` instances
        """

        uv = np.unique(self.df[mark])
        pps = [self.df[self.df[mark] == v] for v in uv]
        names = self.df.columns.values.tolist()
        cnames = self.coord_names
        return[PointPattern(pp, names=names, coord_names=cnames) for pp in pps]

    def unique(self):
        """ Remove duplicate points in the point pattern.

        Two points in a point pattern are deemed to be identical if their
        coordinates are the same, and their marks are the same (if any)

        Returns
        -------
        pp: list
            A deduplicated :class:`PointPattern` instance

        Examples
        --------
        >>> from pointpats import PointPattern
        >>> points = [[1.2, 2.1], [1.2, 2.1], [0, 1], [1, 2]]
        >>> pp = PointPattern(points)
        >>> pp.unique().df
             x    y
        0  1.2  2.1
        2  0.0  1.0
        3  1.0  2.0
        """
        names = self.df.columns.values.tolist()
        coord_names = self.coord_names
        window = self.set_window
        unique_df = self.df.drop_duplicates()
        return PointPattern(unique_df, names=names, coord_names=coord_names,
                            window=window)

    def superimpose(self, point_pattern):
        """Returns a superimposed point pattern.

        Parameters
        ----------
        point_pattern:
              :class:`PointPattern` instance

        Returns
        -------
        superimposed :
            :class:`PointPattern` instance

        Examples
        --------
        >>> from pointpats import PointPattern
        >>> points1 = [[1, 3], [4, 5], [0, 0]]
        >>> points2 = [[5, 6], [1, 4], [0, 0]]
        >>> pp1 = PointPattern(points1)
        >>> pp2 = PointPattern(points2)
        >>> pp1.superimpose(pp2).points
           x  y
        0  1  3
        1  4  5
        2  0  0
        0  5  6
        1  1  4
        """
        names_pp1 = self.df.columns.values.tolist()
        cnames_pp1 = self.coord_names
        names_pp2 = point_pattern.df.columns.values.tolist()
        cnames_pp2 = point_pattern.coord_names
        if names_pp1 != names_pp2 or cnames_pp1 != cnames_pp2:
            raise TypeError('Both point patterns should have similar\
                            attributes and spatial coordinates ')
        pp = pd.concat((self.df, point_pattern.df))
        pp = pp.drop_duplicates()
        return PointPattern(pp, names=names_pp1, coord_names=cnames_pp1)

    def flip_coordinates(self):
        """ Flips the coordinates of a point pattern.

        Doesn't change the structure of data frame. This function swaps
        `_x` and `_y` variables, which are used to represent coordinates.
        """
        self._x, self._y = self._y, self._x

    # Pandas facade
    def _facade(self):
            self.head = self.df.head
            self.tail = self.df.tail
