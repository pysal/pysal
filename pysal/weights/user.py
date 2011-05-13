"""
Convenience functions for the construction of spatial weights based on
contiguity and distance criteria

"""

__author__  = "Sergio J. Rey <srey@asu.edu> "
__all__ = ['queen_from_shapefile', 'rook_from_shapefile', 'knnW_from_array','knnW_from_shapefile', 'threshold_binaryW_from_array', 'threshold_binaryW_from_shapefile', 'threshold_continuousW_from_array', 'threshold_continuousW_from_shapefile', 'kernelW', 'kernelW_from_shapefile', 'adaptive_kernelW', 'adaptive_kernelW_from_shapefile', 'min_threshold_dist_from_shapefile','build_lattice_shapefile']

import pysal
from Contiguity import buildContiguity
from Distance import knnW, Kernel, DistanceBand
from util import get_ids, get_points_array_from_shapefile, min_threshold_distance


def queen_from_shapefile(shapefile, idVariable=None):
    """
    Queen contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile   : string
                  name of polygon shapefile including suffix.
    idVariable  : string
                  name of a column in the shapefile's DBF to use for ids.

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> wq=queen_from_shapefile("../examples/columbus.shp")
    >>> wq.pct_nonzero
    0.098292378175760101
    >>> wq=queen_from_shapefile("../examples/columbus.shp","POLYID")
    >>> wq.pct_nonzero
    0.098292378175760101


    Notes
    -----

    Queen contiguity defines as neighbors any pair of polygons that share at
    least one vertex in their polygon definitions.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return buildContiguity(pysal.open(shapefile), criterion='queen', ids=ids)
    return buildContiguity(pysal.open(shapefile), criterion='queen')

def rook_from_shapefile(shapefile, idVariable=None):
    """
    Rook contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile : string
                name of polygon shapefile including suffix.

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wr=rook_from_shapefile("../examples/columbus.shp", "POLYID")
    >>> wr.pct_nonzero
    0.083298625572678045

    Notes
    -----

    Rook contiguity defines as neighbors any pair of polygons that share a
    common edge in their polygon definitions.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return buildContiguity(pysal.open(shapefile), criterion='rook', ids=ids)
    return buildContiguity(pysal.open(shapefile), criterion='rook')


def spw_from_gal(galfile):
    """
    Sparse scipy matrix for w from a gal file

    Parameters
    ----------

    galfile: string
             name of gal file including suffix

    Returns
    -------

    spw      : scipy sparse matrix in CSR format

    ids      : array
               identifiers for rows/cols of spw

    Examples
    --------

    >>> spw, ids = pysal.weights.user.spw_from_gal("../examples/sids2.gal")
    >>> spw.nnz
    462
    >>> ids
    array(['37001', '37003', '37005', '37007', '37009', '37011', '37013',
           '37015', '37017', '37019', '37021', '37023', '37025', '37027',
           '37029', '37031', '37033', '37035', '37037', '37039', '37041',
           '37043', '37045', '37047', '37049', '37051', '37053', '37055',
           '37057', '37059', '37061', '37063', '37065', '37067', '37069',
           '37071', '37073', '37075', '37077', '37079', '37081', '37083',
           '37085', '37087', '37089', '37091', '37093', '37095', '37097',
           '37099', '37101', '37103', '37105', '37107', '37109', '37111',
           '37113', '37115', '37117', '37119', '37121', '37123', '37125',
           '37127', '37129', '37131', '37133', '37135', '37137', '37139',
           '37141', '37143', '37145', '37147', '37149', '37151', '37153',
           '37155', '37157', '37159', '37161', '37163', '37165', '37167',
           '37169', '37171', '37173', '37175', '37177', '37179', '37181',
           '37183', '37185', '37187', '37189', '37191', '37193', '37195',
           '37197', '37199'], 
          dtype='|S5')
    """

    return pysal.open(galfile,'r').read(sparse=True)

# Distance based weights

def knnW_from_array(array, k=2, p=2, ids=None):
    """
    Nearest neighbor weights from a numpy array

    Parameters
    ----------

    data       : array (n,m)
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

    Returns
    -------

    w         : W instance
                Weights object with binary weights


    Examples
    --------
    >>> import numpy as np
    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wnn2=knnW_from_array(data,k=2)
    >>> wnn4=knnW_from_array(data,k=4)
    >>> wnn4.neighbors[0]
    [1, 5, 6, 2]
    >>> wnn4.neighbors[5]
    [0, 6, 10, 1]
    >>> wnn2.neighbors[0]
    [1, 5]
    >>> wnn2.neighbors[5]
    [0, 6]
    >>> wnn2.pct_nonzero
    0.080000000000000002
    >>> wnn4.pct_nonzero
    0.16
    >>> wnn4=knnW_from_array(data,k=4)
    >>> wnn4.neighbors[0]
    [1, 5, 6, 2]
    >>> wnn4=knnW_from_array(data,k=4)
    >>> wnn3e=knnW(data,p=2,k=3)
    >>> wnn3e.neighbors[0]
    [1, 5, 6]
    >>> wnn3m=knnW(data,p=1,k=3)
    >>> wnn3m.neighbors[0]
    [1, 5, 2]


    Notes
    -----

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    :class:`pysal.weights.W`

    """
    return knnW(array,k=k,p=p,ids=ids)

def knnW_from_shapefile(shapefile,k=2,p=2,idVariable=None):
    """
    Nearest neighbor weights from a shapefile

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

    Returns
    -------

    w         : W instance
                Weights object with binary weights


    Examples
    --------

    Polygon shapefile

    >>> wc=knnW_from_shapefile('../examples/columbus.shp')
    >>> wc.pct_nonzero
    0.040816326530612242
    >>> wc3=knnW_from_shapefile('../examples/columbus.shp',k=3,idVariable="POLYID")
    >>> wc3.weights[1]
    [1, 1, 1]
    >>> wc3.neighbors[1]
    [3, 2, 4]
    >>> wc.neighbors[0]
    [2, 1]

    Point shapefile

    >>> w=knnW_from_shapefile('../examples/juvenile.shp')
    >>> w.pct_nonzero
    0.011904761904761904
    >>> w1=knnW_from_shapefile('../examples/juvenile.shp',k=1)
    >>> w1.pct_nonzero
    0.0059523809523809521
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
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return knnW(data,k=k,p=p,ids=ids)
    return knnW(data,k=k,p=p)

def threshold_binaryW_from_array(array,threshold,p=2):
    """
    Binary weights based on a distance threshold

    Parameters
    ----------

    array       : array (n,m)
                 attribute data, n observations on m attributes
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance

    Returns
    -------

    w         : W instance
                Weights object with binary weights

    Examples
    --------
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> w=threshold_binaryW_from_array(points,threshold=11.2)
    >>> w.weights
    {0: [1, 1], 1: [1, 1], 2: [], 3: [1, 1], 4: [1], 5: [1]}
    >>> w.neighbors
    {0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]}
    >>> 
    """
    return DistanceBand(array,threshold=threshold,p=p)

def threshold_binaryW_from_shapefile(shapefile,threshold,p=2,idVariable=None):
    """
    Threshold distance based binary weights from a shapefile

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

    Returns
    -------

    w         : W instance
                Weights object with binary weights

    Examples
    --------
    >>> w = threshold_binaryW_from_shapefile('../examples/columbus.shp',0.62,idVariable="POLYID")
    >>> w.weights[1]
    [1, 1]
    """
    data = get_points_array_from_shapefile(shapefile)
    if idVariable:
        ids = get_ids(shapefile,idVariable)
        return DistanceBand(data,threshold=threshold,p=p,ids=ids)
    return threshold_binaryW_from_array(data,threshold,p=p)

def threshold_continuousW_from_array(array,threshold,p=2,
                                     alpha=-1):

    """
    Continuous weights based on a distance threshold


    Parameters
    ----------

    array      : array (n,m)
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

    Returns
    -------

    w         : W instance
                Weights object with continuous weights


    Examples
    --------

    inverse distance weights

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> wid=threshold_continuousW_from_array(points,11.2)
    >>> wid.weights[0]
    [0.10000000000000001, 0.089442719099991588]

    gravity weights

    >>> wid2=threshold_continuousW_from_array(points,11.2,alpha=-2.0)
    >>> wid2.weights[0]
    [0.01, 0.0079999999999999984]


    """

    w=DistanceBand(array,threshold=threshold,p=p,alpha=alpha,binary=False)
    return w

def threshold_continuousW_from_shapefile(shapefile,threshold,p=2,
                                         alpha=-1,idVariable=None):
    """
    Threshold distance based continuous weights from a shapefile

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

    Returns
    -------

    w         : W instance
                Weights object with continuous weights

    Examples
    --------
    >>> w = threshold_continuousW_from_shapefile('../examples/columbus.shp',0.62,idVariable="POLYID")
    >>> w.weights[1]
    [1.6702346893743276, 1.7250729841938044]
    """
    data = get_points_array_from_shapefile(shapefile)
    if idVariable:
        ids = get_ids(shapefile,idVariable)
        return DistanceBand(data,threshold=threshold,p=p,alpha=alpha,binary=False,ids=ids)
    return threshold_continuousW_from_array(data,threshold,p=p,alpha=alpha)

# Kernel Weights

def kernelW(points,k=2,function='triangular'):
    """
    Kernel based weights
 
    Parameters
    ----------

    points      : array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. Bandwidth taken as :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
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

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=kernelW(points)
    >>> kw.weights[0]
    [1.0, 0.50000004999999503, 0.44098306152674649]
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
    """
    return Kernel(points,function=function,k=k)

def kernelW_from_shapefile(shapefile,k=2,function='triangular',idVariable=None):
    """
    Kernel based weights
 
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
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
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

    Returns
    -------

    w            : W
                   instance of spatial weights

    Examples
    --------
    >>> kw = kernelW_from_shapefile('../examples/columbus.shp',idVariable='POLYID')
    >>> kw.weights[1]
    [0.20524787824004365, 0.0070787731484506233, 1.0, 0.23051223027663015]
    >>> kw.bandwidth[:3]
    array([[ 0.75333961],
           [ 0.75333961],
           [ 0.75333961]])
    """
    points = get_points_array_from_shapefile(shapefile)
    if idVariable:
        ids = get_ids(shapefile,idVariable)
        return Kernel(points,function=function,k=k,ids=ids)
    return kernelW(points,k=k,function=function)

def adaptive_kernelW(points, bandwidths=None, k=2, function='triangular'):
    """
    Kernel weights with adaptive bandwidths

 
    Parameters
    ----------

    points      : array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidths  : float or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel. 
                  if no bandwidth is specified k is used to determine the
                  adaptive bandwidth
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
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
    [1.0, 0.59999999999999998, 0.55278640450004202, 0.10557280900008403]
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
    [1.0, 0.10557289844279438, 9.9999990066379496e-08]
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
    [0.3989422804014327, 0.26741902915776961, 0.24197074871621341]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])
    """
    return Kernel(points, bandwidth=bandwidths,fixed=False, k=k, function=function)

def adaptive_kernelW_from_shapefile(shapefile, bandwidths=None, k=2, function='triangular', 
                                    idVariable=None):
    """
    Kernel weights with adaptive bandwidths
 
    Parameters
    ----------

    shapefile   : string
                  shapefile name with shp suffix
    bandwidths  : float or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel. 
                  if no bandwidth is specified k is used to determine the
                  adaptive bandwidth
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
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

    Returns
    -------

    w            : W
                   instance of spatial weights


    Examples
    --------
    >>> kwa = adaptive_kernelW_from_shapefile('../examples/columbus.shp')
    >>> kwa.weights[0]
    [1.0, 0.031789067677363891, 9.9999990066379496e-08]
    >>> kwa.bandwidth[:3]
    array([[ 0.59871832],
           [ 0.59871832],
           [ 0.56095647]])
    """
    points = get_points_array_from_shapefile(shapefile)
    if idVariable:
        ids = get_ids(shapefile, idVariable)
        return Kernel(points,bandwidth=bandwidths,fixed=False,k=k,function=function,ids=ids)
    return adaptive_kernelW(points,bandwidths=bandwidths,k=k,function=function)

def min_threshold_dist_from_shapefile(shapefile):
    """
    Kernel weights with adaptive bandwidths
 
    Parameters
    ----------

    shapefile   : string
                  shapefile name with shp suffix

    Returns
    -------
    d            : float
                   minimum nearest neighbor distance between the n observations


    Examples
    --------
    >>> md = min_threshold_dist_from_shapefile('../examples/columbus.shp')
    >>> md
    0.61886415807685413
    """
    points = get_points_array_from_shapefile(shapefile)
    return min_threshold_distance(points)

def build_lattice_shapefile(nrows, ncols, outFileName):
    """
    Build a lattice shapefile with nrows rows and ncols cols

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
        raise ValueError, "outFileName must end with .shp"
    o = pysal.open(outFileName,'w')
    for i in xrange(nrows):
        for j in xrange(ncols):
            ll = i,j
            ul = i,j+1
            ur = i+1,j+1
            lr = i+1,j
            o.write(pysal.cg.Polygon([ll,ul,ur,lr,ll]))
    o.close()

def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
