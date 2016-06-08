"""
Convenience functions for the construction of spatial weights based on
contiguity and distance criteria.
"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

import pysal
from Contiguity import buildContiguity
from Distance import knnW, Kernel, DistanceBand
from util import get_ids, get_points_array_from_shapefile, min_threshold_distance
import numpy as np

__all__ = ['queen_from_shapefile', 'rook_from_shapefile', 'knnW_from_array', 'knnW_from_shapefile', 'threshold_binaryW_from_array', 'threshold_binaryW_from_shapefile', 'threshold_continuousW_from_array', 'threshold_continuousW_from_shapefile', 'kernelW', 'kernelW_from_shapefile', 'adaptive_kernelW', 'adaptive_kernelW_from_shapefile', 'min_threshold_dist_from_shapefile', 'build_lattice_shapefile']


def queen_from_shapefile(shapefile, idVariable=None, sparse=False):
    """
    Queen contiguity weights from a polygon shapefile.

    Parameters
    ----------

    shapefile   : string
                  name of polygon shapefile including suffix.
    idVariable  : string
                  name of a column in the shapefile's DBF to use for ids.
    sparse    : boolean
                If True return WSP instance
                If False return W instance
    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> wq=queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> "%.3f"%wq.pct_nonzero
    '9.829'
    >>> wq=queen_from_shapefile(pysal.examples.get_path("columbus.shp"),"POLYID")
    >>> "%.3f"%wq.pct_nonzero
    '9.829'
    >>> wq=queen_from_shapefile(pysal.examples.get_path("columbus.shp"), sparse=True)
    >>> pct_sp = wq.sparse.nnz *1. / wq.n**2
    >>> "%.3f"%pct_sp
    '0.098'

    Notes
    -----

    Queen contiguity defines as neighbors any pair of polygons that share at
    least one vertex in their polygon definitions.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    shp = pysal.open(shapefile)
    w = buildContiguity(shp, criterion='queen')
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        w.remap_ids(ids)
    else:
        ids = None
    shp.close()
    w.set_shapefile(shapefile, idVariable)

    if sparse:
        w = pysal.weights.WSP(w.sparse, id_order=ids)

    return w


def rook_from_shapefile(shapefile, idVariable=None, sparse=False):
    """
    Rook contiguity weights from a polygon shapefile.

    Parameters
    ----------

    shapefile : string
                name of polygon shapefile including suffix.
    sparse    : boolean
                If True return WSP instance
                If False return W instance

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wr=rook_from_shapefile(pysal.examples.get_path("columbus.shp"), "POLYID")
    >>> "%.3f"%wr.pct_nonzero
    '8.330'
    >>> wr=rook_from_shapefile(pysal.examples.get_path("columbus.shp"), sparse=True)
    >>> pct_sp = wr.sparse.nnz *1. / wr.n**2
    >>> "%.3f"%pct_sp
    '0.083'

    Notes
    -----

    Rook contiguity defines as neighbors any pair of polygons that share a
    common edge in their polygon definitions.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    shp = pysal.open(shapefile)
    w = buildContiguity(shp, criterion='rook')
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        w.remap_ids(ids)
    else:
        ids = None
    shp.close()
    w.set_shapefile(shapefile, idVariable)

    if sparse:
        w = pysal.weights.WSP(w.sparse, id_order=ids)


    return w


def spw_from_gal(galfile):
    """
    Sparse scipy matrix for w from a gal file.

    Parameters
    ----------

    galfile: string
             name of gal file including suffix

    Returns
    -------

    spw      : sparse_matrix
               scipy sparse matrix in CSR format

    ids      : array
               identifiers for rows/cols of spw

    Examples
    --------

    >>> spw = pysal.weights.user.spw_from_gal(pysal.examples.get_path("sids2.gal"))
    >>> spw.sparse.nnz
    462

    """

    return pysal.open(galfile, 'r').read(sparse=True)

# Distance based weights


def knnW_from_array(array, k=2, p=2, ids=None, radius=None):
    """
    Nearest neighbor weights from a numpy array.

    Parameters
    ----------

    data       : array
                 (n,m)
                 attribute data, n observations on m attributes
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

    w         : W
                instance; Weights object with binary weights.

    Examples
    --------
    >>> import numpy as np
    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wnn2=knnW_from_array(data,k=2)
    >>> wnn4=knnW_from_array(data,k=4)
    >>> set([1, 5, 6, 2]) == set(wnn4.neighbors[0])
    True
    >>> set([0, 1, 10, 6]) == set(wnn4.neighbors[5])
    True
    >>> set([1, 5]) == set(wnn2.neighbors[0])
    True
    >>> set([0,6]) == set(wnn2.neighbors[5])
    True
    >>> "%.2f"%wnn2.pct_nonzero
    '8.00'
    >>> wnn4.pct_nonzero
    16.0
    >>> wnn4=knnW_from_array(data,k=4)
    >>> set([ 1,5,6,2]) == set(wnn4.neighbors[0])
    True

    Notes
    -----

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    if radius is not None:
        kdtree = pysal.cg.KDTree(array, distance_metric='Arc', radius=radius)
    else:
        kdtree = pysal.cg.KDTree(array)
    return knnW(kdtree, k=k, p=p, ids=ids)


def knnW_from_shapefile(shapefile, k=2, p=2, idVariable=None, radius=None):
    """
    Nearest neighbor weights from a shapefile.

    Parameters
    ----------

    shapefile  : string
                 shapefile name with shp suffix
    k          : int
                 number of nearest neighbors
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    idVariable : string
                 name of a column in the shapefile's DBF to use for ids
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.

    Returns
    -------

    w         : W
                instance; Weights object with binary weights

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
    '0.595'
    >>>

    Notes
    -----

    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    :class:`pysal.weights.W`

    """

    data = get_points_array_from_shapefile(shapefile)

    if radius is not None:
        kdtree = pysal.cg.KDTree(data, distance_metric='Arc', radius=radius)
    else:
        kdtree = pysal.cg.KDTree(data)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return knnW(kdtree, k=k, p=p, ids=ids)
    return knnW(kdtree, k=k, p=p)


def threshold_binaryW_from_array(array, threshold, p=2, radius=None):
    """
    Binary weights based on a distance threshold.

    Parameters
    ----------

    array       : array
                  (n,m)
                  attribute data, n observations on m attributes
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.

    Returns
    -------

    w         : W
                instance
                Weights object with binary weights

    Examples
    --------
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wcheck = pysal.W({0: [1, 3], 1: [0, 3, ], 2: [], 3: [1, 0], 4: [5], 5: [4]})
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> w=threshold_binaryW_from_array(points,threshold=11.2)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> pysal.weights.util.neighbor_equality(w, wcheck)
    True

    >>>
    """
    if radius is not None:
        array = pysal.cg.KDTree(array, distance_metric='Arc', radius=radius)
    return DistanceBand(array, threshold=threshold, p=p)


def threshold_binaryW_from_shapefile(shapefile, threshold, p=2, idVariable=None, radius=None):
    """
    Threshold distance based binary weights from a shapefile.

    Parameters
    ----------

    shapefile  : string
                 shapefile name with shp suffix
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    idVariable : string
                 name of a column in the shapefile's DBF to use for ids
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.

    Returns
    -------

    w         : W
                instance
                Weights object with binary weights

    Examples
    --------
    >>> w = threshold_binaryW_from_shapefile(pysal.examples.get_path("columbus.shp"),0.62,idVariable="POLYID")
    >>> w.weights[1]
    [1, 1]

    Notes
    -----
    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    """

    data = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        data = pysal.cg.KDTree(data, distance_metric='Arc', radius=radius)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        w = DistanceBand(data, threshold=threshold, p=p)
        w.remap_ids(ids)
        return w
    return threshold_binaryW_from_array(data, threshold, p=p)


def threshold_continuousW_from_array(array, threshold, p=2,
                                     alpha=-1, radius=None):

    """
    Continuous weights based on a distance threshold.

    Parameters
    ----------

    array      : array
                 (n,m)
                 attribute data, n observations on m attributes
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    alpha      : float
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance.
    radius     : If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.

    Returns
    -------

    w         : W
                instance; Weights object with continuous weights.

    Examples
    --------

    inverse distance weights

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wid=threshold_continuousW_from_array(points,11.2)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> wid.weights[0]
    [0.10000000000000001, 0.089442719099991588]

    gravity weights

    >>> wid2=threshold_continuousW_from_array(points,11.2,alpha=-2.0)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [2]
    >>> wid2.weights[0]
    [0.01, 0.0079999999999999984]

    """

    if radius is not None:
        array = pysal.cg.KDTree(array, distance_metric='Arc', radius=radius)
    w = DistanceBand(
        array, threshold=threshold, p=p, alpha=alpha, binary=False)
    return w


def threshold_continuousW_from_shapefile(shapefile, threshold, p=2,
                                         alpha=-1, idVariable=None, radius=None):
    """
    Threshold distance based continuous weights from a shapefile.

    Parameters
    ----------

    shapefile  : string
                 shapefile name with shp suffix
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    alpha      : float
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance.
    idVariable : string
                 name of a column in the shapefile's DBF to use for ids
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.

    Returns
    -------

    w         : W
                instance; Weights object with continuous weights.

    Examples
    --------
    >>> w = threshold_continuousW_from_shapefile(pysal.examples.get_path("columbus.shp"),0.62,idVariable="POLYID")
    >>> w.weights[1]
    [1.6702346893743334, 1.7250729841938093]

    Notes
    -----
    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    """

    data = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        data = pysal.cg.KDTree(data, distance_metric='Arc', radius=radius)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        w = DistanceBand(data, threshold=threshold, p=p, alpha=alpha, binary=False)
        w.remap_ids(ids)
    else:
        w =  threshold_continuousW_from_array(data, threshold, p=p, alpha=alpha)
    w.set_shapefile(shapefile,idVariable)
    return w


# Kernel Weights


def kernelW(points, k=2, function='triangular', fixed=True,
        radius=None, diagonal=False):
    """
    Kernel based weights.

    Parameters
    ----------

    points      : array
                  (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. Bandwidth taken as :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).
    function    : {'triangular','uniform','quadratic','epanechnikov','quartic','bisquare','gaussian'}
                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = |z| \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  epanechnikov

                  .. math::

                      K(z) = (1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  bisquare

                  .. math::

                      K(z) = (1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    fixed        : boolean
                   If true then :math:`h_i=h \\forall i`. If false then
                   bandwidth is adaptive across observations.
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.
    diagonal   : boolean
                 If true, set diagonal weights = 1.0, if false (default)
                 diagonal weights are set to value according to kernel
                 function

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=kernelW(points)
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

    use different k

    >>> kw=kernelW(points,k=3)
    >>> kw.neighbors[0]
    [0, 1, 3, 4]
    >>> kw.bandwidth
    array([[ 22.36068201],
           [ 22.36068201],
           [ 22.36068201],
           [ 22.36068201],
           [ 22.36068201],
           [ 22.36068201]])

    Diagonals to 1.0

    >>> kq = kernelW(points,function='gaussian')
    >>> kq.weights
    {0: [0.3989422804014327, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 0.3989422804014327, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 0.3989422804014327, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 0.3989422804014327, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 0.3989422804014327, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 0.3989422804014327]}
    >>> kqd = kernelW(points, function='gaussian', diagonal=True)
    >>> kqd.weights
    {0: [1.0, 0.35206533556593145, 0.3412334260702758], 1: [0.35206533556593145, 1.0, 0.2419707487162134, 0.3412334260702758, 0.31069657591175387], 2: [0.2419707487162134, 1.0, 0.31069657591175387], 3: [0.3412334260702758, 0.3412334260702758, 1.0, 0.3011374490937829, 0.26575287272131043], 4: [0.31069657591175387, 0.31069657591175387, 0.3011374490937829, 1.0, 0.35206533556593145], 5: [0.26575287272131043, 0.35206533556593145, 1.0]}

    """

    if radius is not None:
        points = pysal.cg.KDTree(points, distance_metric='Arc', radius=radius)
    return Kernel(points, function=function, k=k, fixed=fixed,
            diagonal=diagonal)


def kernelW_from_shapefile(shapefile, k=2, function='triangular',
        idVariable=None, fixed=True, radius=None, diagonal=False):
    """
    Kernel based weights.

    Parameters
    ----------

    shapefile   : string
                  shapefile name with shp suffix
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. Bandwidth taken as :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).
    function    : {'triangular','uniform','quadratic','epanechnikov', 'quartic','bisquare','gaussian'}

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = |z| \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  epanechnikov

                  .. math::

                      K(z) = (1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  bisquare

                  .. math::

                      K(z) = (1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)
    idVariable   : string
                   name of a column in the shapefile's DBF to use for ids

    fixed        : binary
                   If true then :math:`h_i=h \\forall i`. If false then
                   bandwidth is adaptive across observations.
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.
    diagonal   : boolean
                 If true, set diagonal weights = 1.0, if false (default)
                 diagonal weights are set to value according to kernel
                 function

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> kw = pysal.kernelW_from_shapefile(pysal.examples.get_path("columbus.shp"),idVariable='POLYID', function = 'gaussian')

    >>> kwd = pysal.kernelW_from_shapefile(pysal.examples.get_path("columbus.shp"),idVariable='POLYID', function = 'gaussian', diagonal = True)
    >>> set(kw.neighbors[1]) == set([4, 2, 3, 1])
    True
    >>> set(kwd.neighbors[1]) == set([4, 2, 3, 1])
    True
    >>>
    >>> set(kw.weights[1]) == set( [0.2436835517263174, 0.29090631630909874, 0.29671172124745776, 0.3989422804014327])
    True
    >>> set(kwd.weights[1]) == set( [0.2436835517263174, 0.29090631630909874, 0.29671172124745776, 1.0])
    True


    Notes
    -----
    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    """

    points = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        points = pysal.cg.KDTree(points, distance_metric='Arc', radius=radius)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return Kernel(points, function=function, k=k, ids=ids, fixed=fixed,
                diagonal = diagonal)
    return kernelW(points, k=k, function=function, fixed=fixed,
            diagonal=diagonal)


def adaptive_kernelW(points, bandwidths=None, k=2, function='triangular',
        radius=None, diagonal=False):
    """
    Kernel weights with adaptive bandwidths.

    Parameters
    ----------

    points      : array
                  (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidths  : float
                  or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel.
                  if no bandwidth is specified k is used to determine the
                  adaptive bandwidth
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    function    : {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = |z| \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.
    diagonal   : boolean
                 If true, set diagonal weights = 1.0, if false (default)
                 diagonal weights are set to value according to kernel
                 function
    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------

    User specified bandwidths

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=adaptive_kernelW(points,bandwidths=bw)
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

    >>> kwea=adaptive_kernelW(points)
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

    >>> kweag=adaptive_kernelW(points,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])

    with diagonal

    >>> kweag = pysal.adaptive_kernelW(points, function='gaussian')
    >>> kweagd = pysal.adaptive_kernelW(points, function='gaussian', diagonal=True)
    >>> kweag.neighbors[0]
    [0, 1, 3]
    >>> kweagd.neighbors[0]
    [0, 1, 3]
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweagd.weights[0]
    [1.0, 0.2674190291577696, 0.2419707487162134]

    """
    if radius is not None:
        points = pysal.cg.KDTree(points, distance_metric='Arc', radius=radius)
    return Kernel(points, bandwidth=bandwidths, fixed=False, k=k,
            function=function, diagonal=diagonal)


def adaptive_kernelW_from_shapefile(shapefile, bandwidths=None, k=2, function='triangular',
                                    idVariable=None, radius=None,
                                    diagonal = False):
    """
    Kernel weights with adaptive bandwidths.

    Parameters
    ----------

    shapefile   : string
                  shapefile name with shp suffix
    bandwidths  : float
                  or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel.
                  if no bandwidth is specified k is used to determine the
                  adaptive bandwidth
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    function    : {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform

                  .. math::

                      K(z) = |z| \ if |z| \le 1

                  quadratic

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1

                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)
    idVariable   : string
                   name of a column in the shapefile's DBF to use for ids
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.
    diagonal   : boolean
                 If true, set diagonal weights = 1.0, if false (default)
                 diagonal weights are set to value according to kernel
                 function

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> kwa = pysal.adaptive_kernelW_from_shapefile(pysal.examples.get_path("columbus.shp"), function='gaussian')
    >>> kwad = pysal.adaptive_kernelW_from_shapefile(pysal.examples.get_path("columbus.shp"), function='gaussian', diagonal=True)
    >>> kwa.neighbors[0]
    [0, 2, 1]
    >>> kwad.neighbors[0]
    [0, 2, 1]
    >>> kwa.weights[0]
    [0.3989422804014327, 0.24966013701844503, 0.2419707487162134]
    >>> kwad.weights[0]
    [1.0, 0.24966013701844503, 0.2419707487162134]
    >>>

    Notes
    -----
    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    """
    points = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        points = pysal.cg.KDTree(points, distance_metric='Arc', radius=radius)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return Kernel(points, bandwidth=bandwidths, fixed=False, k=k,
                function=function, ids=ids, diagonal=diagonal)
    return adaptive_kernelW(points, bandwidths=bandwidths, k=k,
            function=function, diagonal=diagonal)


def min_threshold_dist_from_shapefile(shapefile, radius=None, p=2):
    """
    Kernel weights with adaptive bandwidths.

    Parameters
    ----------

    shapefile  : string
                 shapefile name with shp suffix
    radius     : float
                 If supplied arc_distances will be calculated
                 based on the given radius. p will be ignored.
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance

    Returns
    -------
    d            : float
                   minimum nearest neighbor distance between the n observations

    Examples
    --------
    >>> md = min_threshold_dist_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> md
    0.61886415807685413
    >>> min_threshold_dist_from_shapefile(pysal.examples.get_path("stl_hom.shp"), pysal.cg.sphere.RADIUS_EARTH_MILES)
    31.846942936393717

    Notes
    -----
    Supports polygon or point shapefiles. For polygon shapefiles, distance is
    based on polygon centroids. Distances are defined using coordinates in
    shapefile which are assumed to be projected and not geographical
    coordinates.

    """
    points = get_points_array_from_shapefile(shapefile)
    if radius is not None:
        kdt = pysal.cg.kdtree.Arc_KDTree(points, radius=radius)
        nn = kdt.query(kdt.data, k=2)
        nnd = nn[0].max(axis=0)[1]
        return nnd
    return min_threshold_distance(points, p)


def build_lattice_shapefile(nrows, ncols, outFileName):
    """
    Build a lattice shapefile with nrows rows and ncols cols.

    Parameters
    ----------

    nrows       : int
                  Number of rows
    ncols       : int
                  Number of cols
    outFileName : str
                  shapefile name with shp suffix

    Returns
    -------
    None

    """
    if not outFileName.endswith('.shp'):
        raise ValueError("outFileName must end with .shp")
    o = pysal.open(outFileName, 'w')
    dbf_name = outFileName.split(".")[0] + ".dbf"
    d = pysal.open(dbf_name, 'w')
    d.header = [ 'ID' ]
    d.field_spec = [ ('N', 8, 0) ]
    c = 0
    for i in xrange(nrows):
        for j in xrange(ncols):
            ll = i, j
            ul = i, j + 1
            ur = i + 1, j + 1
            lr = i + 1, j
            o.write(pysal.cg.Polygon([ll, ul, ur, lr, ll]))
            d.write([c])
            c += 1
    d.close()
    o.close()

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


