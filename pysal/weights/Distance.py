"""
Distance based spatial weights.

"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

import pysal
import scipy.spatial
from pysal.common import KDTree
from pysal.weights import W
import scipy.stats
import numpy as np
from util import isKDTree

__all__ = ["knnW", "Kernel", "DistanceBand"]

def knnW(data, k=2, p=2, ids=None):
    """
    Creates nearest neighbor weights matrix based on k nearest
    neighbors.

    Parameters
    ----------

    kdtree      : object
                  PySAL KDTree or ArcKDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    k           : int
                  number of nearest neighbors
    p           : float
                  Minkowski p-norm distance metric parameter:
                  1<=p<=infinity
                  2: Euclidean distance
                  1: Manhattan distance
                  Ignored if the KDTree is an ArcKDTree
    ids         : list
                  identifiers to attach to each observation

    Returns
    -------

    w         : W
                instance
                Weights object with binary weights

    Examples
    --------

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kd = pysal.cg.kdtree.KDTree(np.array(points))
    >>> wnn2 = pysal.knnW(kd, 2)
    >>> [1,3] == wnn2.neighbors[0]
    True

    ids

    >>> wnn2 = knnW(kd,2)
    >>> wnn2[0]
    {1: 1.0, 3: 1.0}
    >>> wnn2[1]
    {0: 1.0, 3: 1.0}

    now with 1 rather than 0 offset

    >>> wnn2 = knnW(kd, 2, ids=range(1,7))
    >>> wnn2[1]
    {2: 1.0, 4: 1.0}
    >>> wnn2[2]
    {1: 1.0, 4: 1.0}
    >>> 0 in wnn2.neighbors
    False

    Notes
    -----

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    pysal.weights.W

    """
    if isKDTree(data):
        kdt = data
        data = kdt.data
    else:
        kdt = KDTree(data)
    nnq = kdt.query(data, k=k+1, p=p)
    info = nnq[1]

    neighbors = {}
    for i, row in enumerate(info):
        row = row.tolist()
        if i in row:
            row.remove(i)
            focal = i
        if ids:
            row = [ ids[j] for j in row]
            focal = ids[i]
        neighbors[focal] = row
    return pysal.weights.W(neighbors,  id_order=ids)


class Kernel(W):
    """
    Spatial weights based on kernel functions.

    Parameters
    ----------

    data        : array
                  (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidth   : float
                  or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel.
    fixed       : binary
                  If true then :math:`h_i=h \\forall i`. If false then
                  bandwidth is adaptive across observations.
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    diagonal    : boolean
                  If true, set diagonal weights = 1.0, if false (default),
                  diagonals weights are set to value according to kernel
                  function.
    function    : {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = 1/2 \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    eps         : float
                  adjustment to ensure knn distance range is closed on the
                  knnth observations

    Attributes
    ----------
    weights : dict
              Dictionary keyed by id with a list of weights for each neighbor

    neighbors : dict
                of lists of neighbors keyed by observation id

    bandwidth : array
                array of bandwidths

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=Kernel(points)
    >>> kw.weights[0]
    [1.0, 0.500000049999995, 0.4409830615267465]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> kw.bandwidth
    array([[ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002]])
    >>> kw15=Kernel(points,bandwidth=15.0)
    >>> kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> kw15.neighbors[0]
    [0, 1, 3]
    >>> kw15.bandwidth
    array([[ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.]])

    Adaptive bandwidths user specified

    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=Kernel(points,bandwidth=bw)
    >>> kwa.weights[0]
    [1.0, 0.6, 0.552786404500042, 0.10557280900008403]
    >>> kwa.neighbors[0]
    [0, 1, 3, 4]
    >>> kwa.bandwidth
    array([[ 25. ],
           [ 15. ],
           [ 25. ],
           [ 16. ],
           [ 14.5],
           [ 25. ]])

    Endogenous adaptive bandwidths

    >>> kwea=Kernel(points,fixed=False)
    >>> kwea.weights[0]
    [1.0, 0.10557289844279438, 9.99999900663795e-08]
    >>> kwea.neighbors[0]
    [0, 1, 3]
    >>> kwea.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])

    Endogenous adaptive bandwidths with Gaussian kernel

    >>> kweag=Kernel(points,fixed=False,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])

    Diagonals to 1.0

    >>> kq = Kernel(points,function='gaussian')
    >>> kq.weights
    {0: [0.3989422804014327, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 0.3989422804014327, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 0.3989422804014327, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 0.3989422804014327, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 0.3989422804014327, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 0.3989422804014327]}
    >>> kqd = Kernel(points, function='gaussian', diagonal=True)
    >>> kqd.weights
    {0: [1.0, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 1.0, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 1.0, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 1.0, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 1.0, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 1.0]}

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=2,
                 function='triangular', eps=1.0000001, ids=None,
                 diagonal=False):
        if isKDTree(data):
            self.kdt = data
            self.data = self.kdt.data
            data = self.data
        else:
            self.data = data
            self.kdt = KDTree(self.data)
        self.k = k + 1
        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), 'float') * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()

        self._eval_kernel()
        neighbors, weights = self._k_to_W(ids)
        if diagonal:
            for i in neighbors:
                weights[i][neighbors[i].index(i)] = 1.0
        W.__init__(self, neighbors, weights, ids)

    def _k_to_W(self, ids=None):
        allneighbors = {}
        weights = {}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self.data))
        for i, neighbors in enumerate(self.kernel):
            if len(self.neigh[i]) == 0:
                allneighbors[ids[i]] = []
                weights[ids[i]] = []
            else:
                allneighbors[ids[i]] = list(ids[self.neigh[i]])
                weights[ids[i]] = self.kernel[i].tolist()
        return allneighbors, weights

    def _set_bw(self):
        dmat, neigh = self.kdt.query(self.data, k=self.k)
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(dmat)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)
            # identify knn neighbors for each point
            nnq = self.kdt.query(self.data, k=self.k)
            self.neigh = nnq[1]

    def _eval_kernel(self):
        # get points within bandwidth distance of each point
        if not hasattr(self, 'neigh'):
            kdtq = self.kdt.query_ball_point
            neighbors = [kdtq(self.data[i], r=bwi[0]) for i,
                         bwi in enumerate(self.bandwidth)]
            self.neigh = neighbors
        # get distances for neighbors
        bw = self.bandwidth

        kdtq = self.kdt.query
        z = []
        for i, nids in enumerate(self.neigh):
            di, ni = kdtq(self.data[i], k=len(nids))
            if not isinstance(di, np.ndarray):
                di = np.asarray([di] * len(nids))
                ni = np.asarray([ni] * len(nids))
            zi = np.array([dict(zip(ni, di))[nid] for nid in nids]) / bw[i]
            z.append(zi)
        zs = z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            self.kernel = [1 - zi for zi in zs]
        elif self.function == 'uniform':
            self.kernel = [np.ones(zi.shape) * 0.5 for zi in zs]
        elif self.function == 'quadratic':
            self.kernel = [(3. / 4) * (1 - zi ** 2) for zi in zs]
        elif self.function == 'quartic':
            self.kernel = [(15. / 16) * (1 - zi ** 2) ** 2 for zi in zs]
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            self.kernel = [c * np.exp(-(zi ** 2) / 2.) for zi in zs]
        else:
            print('Unsupported kernel function', self.function)


class DistanceBand(W):
    """
    Spatial weights based on distance band.

    Parameters
    ----------

    data        : array
                  (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    binary     : boolean
                 If true w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
                 If false wij=dij^{alpha}
    alpha      : float
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance. If binary is True, alpha is ignored

    ids         : list
                  values to use for keys of the neighbors and weights dicts

    Attributes
    ----------
    weights : dict
              of neighbor weights keyed by observation id

    neighbors : dict
                of neighbors keyed by observation id

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wcheck = pysal.W({0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]})
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w=DistanceBand(points,threshold=11.2)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> pysal.weights.util.neighbor_equality(w, wcheck)
    True
    >>> w=DistanceBand(points,threshold=14.2)
    >>> wcheck = pysal.W({0: [1, 3], 1: [0, 3, 4], 2: [4], 3: [1, 0], 4: [5, 2, 1], 5: [4]})
    >>> pysal.weights.util.neighbor_equality(w, wcheck)
    True



    inverse distance weights

    >>> w=DistanceBand(points,threshold=11.2,binary=False)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w.weights[0]
    [0.10000000000000001, 0.089442719099991588]
    >>> w.neighbors[0]
    [1, 3]
    >>>

    gravity weights

    >>> w=DistanceBand(points,threshold=11.2,binary=False,alpha=-2.)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w.weights[0]
    [0.01, 0.0079999999999999984]

    Notes
    -----

    This was initially implemented running scipy 0.8.0dev (in epd 6.1).
    earlier versions of scipy (0.7.0) have a logic bug in scipy/sparse/dok.py
    so serge changed line 221 of that file on sal-dev to fix the logic bug.

    """

    def __init__(self, data, threshold, p=2, alpha=-1.0, binary=True, ids=None):
        """Casting to floats is a work around for a bug in scipy.spatial.
        See detail in pysal issue #126.

        """
        if isKDTree(data):
            self.kd = data
            self.data = self.kd.data
        else:
            try:
                data = np.asarray(data)
                if data.dtype.kind != 'f':
                    data = data.astype(float)
                self.data = data
                self.kd = KDTree(self.data)
            except:
                raise ValueError("Could not make array from data")

        self.p = p
        self.threshold = threshold
        self.binary = binary
        self.alpha = alpha
        self._band()
        neighbors, weights = self._distance_to_W(ids)
        W.__init__(self, neighbors, weights, ids)

    def _band(self):
        """Find all pairs within threshold.

        """
        self.dmat = self.kd.sparse_distance_matrix(
                self.kd, max_distance=self.threshold)

    def _distance_to_W(self, ids=None):
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(self.dmat.shape[0])
        neighbors = dict([(i,[]) for i in ids])
        weights = dict([(i,[]) for i in ids])
        if self.binary:
            for key,weight in self.dmat.items():
                i,j = key
                if i != j:
                    if j not in neighbors[i]:
                        weights[i].append(1)
                        neighbors[i].append(j)
                    if i not in neighbors[j]:
                        weights[j].append(1)
                        neighbors[j].append(i)

        else:
            for key,weight in self.dmat.items():
                i,j = key
                if i != j:
                    if j not in neighbors[i]:
                        weights[i].append(weight**self.alpha)
                        neighbors[i].append(j)
                    if i not in neighbors[j]:
                        weights[j].append(weight**self.alpha)
                        neighbors[j].append(i)

        return neighbors, weights


def _test():
    import doctest
    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    #doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':
    _test()
