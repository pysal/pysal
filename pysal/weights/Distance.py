from ..cg.kdtree import KDTree
from .weights import W
from .util import isKDTree, get_ids, get_points_array_from_shapefile, get_points_array
import copy
from warnings import warn as Warn
import numpy as np

__all__ = ["KNN", "Kernel", "DistanceBand"]
__author__ = "Sergio J. Rey <srey@asu.edu>, Levi John Wolf <levi.john.wolf@gmail.com>"

import pysal
import scipy.spatial
from pysal.common import KDTree
from pysal.weights import W, WSP
from pysal.weights.util import WSP2W
import scipy.stats
from scipy.spatial import distance_matrix
import scipy.sparse as sp
import numpy as np
from util import isKDTree

def knnW(data, k=2, p=2, ids=None, radius=None, distance_metric='euclidean'):
    """
    This is deprecated. Use the pysal.weights.KNN class instead. 
    """
    #Warn('This function is deprecated. Please use pysal.weights.KNN', UserWarning)
    return KNN(data, k=k, p=p, ids=ids, radius=radius,
            distance_metric=distance_metric)

class KNN(W):
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
    >>> wnn2 = pysal.KNN(kd, 2)
    >>> [1,3] == wnn2.neighbors[0]
    True

    ids

    >>> wnn2 = KNN(kd,2)
    >>> wnn2[0]
    {1: 1.0, 3: 1.0}
    >>> wnn2[1]
    {0: 1.0, 3: 1.0}

    now with 1 rather than 0 offset

    >>> wnn2 = KNN(kd, 2, ids=range(1,7))
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
    :class:`pysal.weights.W`
    """
    def __init__(self, data, k=2, p=2, ids=None, radius=None, distance_metric='euclidean'):
        if isKDTree(data):
            self.kdtree = data
            self.data = data.data
        else:
            self.data = data
            self.kdtree = KDTree(data, radius=radius, distance_metric=distance_metric)
        self.k = k 
        self.p = p
        this_nnq = self.kdtree.query(self.data, k=k+1, p=p)
        
        to_weight = this_nnq[1]
        if ids is None:
            ids = list(range(to_weight.shape[0]))
        
        neighbors = {}
        for i,row in enumerate(to_weight):
            row = row.tolist()
            row.remove(i)
            row = [ids[j] for j in row]
            focal = ids[i]
            neighbors[focal] = row
        W.__init__(self, neighbors, id_order=ids)
    
    @classmethod
    def from_shapefile(cls, filepath, **kwargs):
        """
        Nearest neighbor weights from a shapefile.

        Parameters
        ----------

        data       : string
                     shapefile containing attribute data.
        k          : int
                     number of nearest neighbors
        p          : float
                     Minkowski p-norm distance metric parameter:
                     1<=p<=infinity
                     2: Euclidean distance
                     1: Manhattan distance
        ids        : list
                     identifiers to attach to each observation
        radius     : float
                     If supplied arc_distances will be calculated
                     based on the given radius. p will be ignored.

        Returns
        -------

        w         : KNN
                    instance; Weights object with binary weights.

        Examples
        --------

        Polygon shapefile

        >>> wc=knnW_from_shapefile(pysal.examples.get_path("columbus.shp"))
        >>> "%.4f"%wc.pct_nonzero
        '4.0816'
        >>> set([2,1]) == set(wc.neighbors[0])
        True
        >>> wc3=pysal.knnW_from_shapefile(pysal.examples.get_path("columbus.shp"),k=3)
        >>> set(wc3.neighbors[0]) == set([2,1,3])
        True
        >>> set(wc3.neighbors[2]) == set([4,3,0])
        True

        1 offset rather than 0 offset

        >>> wc3_1=knnW_from_shapefile(pysal.examples.get_path("columbus.shp"),k=3,idVariable="POLYID")
        >>> set([4,3,2]) == set(wc3_1.neighbors[1])
        True
        >>> wc3_1.weights[2]
        [1.0, 1.0, 1.0]
        >>> set([4,1,8]) == set(wc3_1.neighbors[2])
        True


        Point shapefile

        >>> w=knnW_from_shapefile(pysal.examples.get_path("juvenile.shp"))
        >>> w.pct_nonzero
        1.1904761904761905
        >>> w1=knnW_from_shapefile(pysal.examples.get_path("juvenile.shp"),k=1)
        >>> "%.3f"%w1.pct_nonzero

        Notes
        -----

        Ties between neighbors of equal distance are arbitrarily broken.

        See Also
        --------
        :class:`pysal.weights.KNN`
        :class:`pysal.weights.W`
        """
        return cls(get_points_array_from_shapefile(filepath), **kwargs)
    
    @classmethod
    def from_array(cls, array, **kwargs):
        """
        Creates nearest neighbor weights matrix based on k nearest
        neighbors.

        Parameters
        ----------
        array       : np.ndarray
                      (n, k) array representing n observations on 
                      k characteristics used to measure distances 
                      between the n objects
        **kwargs    : keyword arguments, see Rook

        Returns
        -------
        w         : W
                    instance
                    Weights object with binary weights

        Examples
        --------
        >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
        >>> wnn2 = pysal.KNN.from_array(points, 2)
        >>> [1,3] == wnn2.neighbors[0]
        True

        ids

        >>> wnn2 = KNN.from_array(points,2)
        >>> wnn2[0]
        {1: 1.0, 3: 1.0}
        >>> wnn2[1]
        {0: 1.0, 3: 1.0}

        now with 1 rather than 0 offset

        >>> wnn2 = KNN.from_array(points, 2, ids=range(1,7))
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
        :class: `pysal.weights.KNN`
        :class:`pysal.weights.W`
        """
        return cls(array, **kwargs)

    @classmethod
    def from_dataframe(cls, df, geom_col='geometry', ids=None, **kwargs):
        """
        Make KNN weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :   string
                    column name of the geometry stored in df
        ids     :   string or iterable
                    if string, the column name of the indices from the dataframe
                    if iterable, a list of ids to use for the W
                    if None, df.index is used.

        See Also
        --------
        :class: `pysal.weights.KNN`
        :class:`pysal.weights.W`
        """
        pts = get_points_array(df[geom_col])
        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, ids=ids, **kwargs)

    def reweight(self, k=None, p=None, new_data=None, new_ids=None, inplace=True):
        """
        Redo K-Nearest Neighbor weights construction using given parameters

        Parameters
        ----------
        new_data    : np.ndarray
                      an array containing additional data to use in the KNN
                      weight
        new_ids     : list
                      a list aligned with new_data that provides the ids for
                      each new observation
        inplace     : bool
                      a flag denoting whether to modify the KNN object 
                      in place or to return a new KNN object
        k           : int
                      number of nearest neighbors
        p           : float
                      Minkowski p-norm distance metric parameter:
                      1<=p<=infinity
                      2: Euclidean distance
                      1: Manhattan distance
                      Ignored if the KDTree is an ArcKDTree

        Returns
        -------
        A copy of the object using the new parameterization, or None if the
        object is reweighted in place.
        """
        if (new_data is not None):
            new_data = np.asarray(new_data).reshape(-1,2)
            data = np.vstack((self.data, new_data)).reshape(-1,2)
            if new_ids is not None:
                ids = copy.deepcopy(self.id_order)
                ids.extend(list(new_ids))
            else:
                ids = list(range(data.shape[0]))
        elif (new_data is None) and (new_ids is None):
            # If not, we can use the same kdtree we have
            data = self.kdtree
            ids = self.id_order
        elif (new_data is None) and (new_ids is not None):
            Warn('Remapping ids must be done using w.remap_ids')
        if k is None:
            k = self.k
        if p is None:
            p = self.p
        if inplace:
            self._reset()
            self.__init__(data, ids=ids, k=k, p=p)
        else:
            return KNN(data, ids=ids, k=k, p=p)

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
    
    @classmethod
    def from_shapefile(cls, filepath, idVariable=None,  **kwargs):
        """
        Kernel based weights from shapefile

        Arguments
        ---------
        shapefile   : string
                      shapefile name with shp suffix
        idVariable  : string
                      name of column in shapefile's DBF to use for ids

        Returns
        --------
        Kernel Weights Object

        See Also
        ---------
        :class:`pysal.weights.Kernel`
        :class:`pysal.weights.W`
        """
        points = get_points_array_from_shapefile(filepath)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        return cls.from_array(points, ids=ids, **kwargs)
    
    @classmethod
    def from_array(cls, array, **kwargs):
        """
        Construct a Kernel weights from an array. Supports all the same options
        as :class:`pysal.weights.Kernel`

        See Also
        --------
        :class:`pysal.weights.Kernel`
        :class:`pysal.weights.W`
        """
        return cls(array, **kwargs)

    @classmethod
    def from_dataframe(cls, df, geom_col='geometry', ids=None, **kwargs):
        """
        Make Kernel weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :   string
                    column name of the geometry stored in df
        ids     :   string or iterable
                    if string, the column name of the indices from the dataframe
                    if iterable, a list of ids to use for the W
                    if None, df.index is used.

        See Also
        --------
        :class:`pysal.weights.Kernel`
        :class:`pysal.weights.W`
        """
        pts = get_points_array(df[geom_col])
        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, ids=ids, **kwargs)

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
    
    build_sp    : boolean
                  True to build sparse distance matrix and false to build dense
                  distance matrix; significant speed gains may be obtained
                  dending on the sparsity of the of distance_matrix and
                  threshold that is applied
    silent      : boolean
                  By default PySAL will print a warning if the
                  dataset contains any disconnected observations or
                  islands. To silence this warning set this
                  parameter to True.

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

    def __init__(self, data, threshold, p=2, alpha=-1.0, binary=True, ids=None,
            build_sp=True, silent=False):
        """Casting to floats is a work around for a bug in scipy.spatial.
        See detail in pysal issue #126.

        """
        self.p = p
        self.threshold = threshold
        self.binary = binary
        self.alpha = alpha
        self.build_sp = build_sp
        self.silent = silent
        
        if isKDTree(data):
            self.kd = data
            self.data = self.kd.data
        else:
            if self.build_sp:
                try:
                    data = np.asarray(data)
                    if data.dtype.kind != 'f':
                        data = data.astype(float)
                    self.data = data
                    self.kd = KDTree(self.data)
                except:
                    raise ValueError("Could not make array from data")        
            else:
                self.data = data
                self.kd = None       
        self._band()
        neighbors, weights = self._distance_to_W(ids)
        W.__init__(self, neighbors, weights, ids, silent_island_warning=self.silent)

    @classmethod
    def from_shapefile(cls, filepath, threshold, idVariable=None, **kwargs):
        """
        Distance-band based weights from shapefile

        Arguments
        ---------
        shapefile   : string
                      shapefile name with shp suffix
        idVariable  : string
                      name of column in shapefile's DBF to use for ids

        Returns
        --------
        Kernel Weights Object

        See Also
        ---------
        :class: `pysal.weights.DistanceBand`
        :class: `pysal.weights.W`
        """
        points = get_points_array_from_shapefile(filepath)
        if idVariable is not None:
            ids = get_ids(filepath, idVariable)
        else:
            ids = None
        return cls.from_array(points, threshold, ids=ids, **kwargs)
    
    @classmethod
    def from_array(cls, array, threshold, **kwargs):
        """
        Construct a DistanceBand weights from an array. Supports all the same options
        as :class:`pysal.weights.DistanceBand`

        See Also
        --------
        :class:`pysal.weights.DistanceBand`
        :class:`pysal.weights.W`
        """
        return cls(array, threshold, **kwargs)
    
    @classmethod
    def from_dataframe(cls, df, threshold, geom_col='geometry', ids=None, **kwargs):
        """
        Make DistanceBand weights from a dataframe.

        Parameters
        ----------
        df      :   pandas.dataframe
                    a dataframe with a geometry column that can be used to
                    construct a W object
        geom_col :   string
                    column name of the geometry stored in df
        ids     :   string or iterable
                    if string, the column name of the indices from the dataframe
                    if iterable, a list of ids to use for the W
                    if None, df.index is used.

        See Also
        --------
        :class:`pysal.weights.DistanceBand`
        :class:`pysal.weights.W`
        """
        pts = get_points_array(df[geom_col])
        if ids is None:
            ids = df.index.tolist()
        elif isinstance(ids, str):
            ids = df[ids].tolist()
        return cls(pts, threshold, ids=ids, **kwargs)

    def _band(self):
        """Find all pairs within threshold.

        """
        if self.build_sp:    
            self.dmat = self.kd.sparse_distance_matrix(
                    self.kd, max_distance=self.threshold).tocsr()
        else:
            if str(self.kd).split('.')[-1][0:10] == 'Arc_KDTree':
            	raise TypeError('Unable to calculate dense arc distance matrix;'
            	        ' parameter "build_sp" must be set to True for arc'
            	        ' distance type weight')
            self.dmat = self._spdistance_matrix(self.data, self.data, self.threshold)


    def _distance_to_W(self, ids=None):
        if self.binary:
            self.dmat[self.dmat>0] = 1
            self.dmat.eliminate_zeros()
            tempW = WSP2W(WSP(self.dmat), silent_island_warning=self.silent)
            neighbors = tempW.neighbors
            weight_keys = tempW.weights.keys()
            weight_vals = tempW.weights.values()
            weights = dict(zip(weight_keys, map(list, weight_vals)))
            return neighbors, weights
        else:
            weighted = self.dmat.power(self.alpha)
            weighted[weighted==np.inf] = 0
            weighted.eliminate_zeros()
            tempW = WSP2W(WSP(weighted), silent_island_warning=self.silent)
            neighbors = tempW.neighbors
            weight_keys = tempW.weights.keys()
            weight_vals = tempW.weights.values()
            weights = dict(zip(weight_keys, map(list, weight_vals)))
            return neighbors, weights

    def _spdistance_matrix(self, x,y, threshold=None):
        dist = distance_matrix(x,y)
        if threshold is not None:
            zeros = dist > threshold
            dist[zeros] = 0
        return sp.csr_matrix(dist)

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
