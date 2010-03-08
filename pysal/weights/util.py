"""
Convenience functions for the construction of spatial weights based on
contiguity and distance criteria

Author(s):
    Serge Rey srey@asu.edu

"""
from Contiguity import buildContiguity
from Distance import knnW
import numpy as np



def queen_from_shapefile(shapefile):
    """
    Queen contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile : string
                name of shapefile including suffix.

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wq=queen_from_shapefile("../examples/columbus.shp")
    >>> wq.pct_nonzero
    0.098292378175760101


    Notes
    -----

    Queen contiguity defines as neighbors any pair of polygons that share at
    least one vertex in their polygon definitions.

    See Also
    --------

    pysal.weights.W
    """
    return buildContiguity(shapefile,criteria='queen')

def rook_from_shapefile(shapefile):
    """
    Rook contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile : string
                 name of shapefile including suffix.

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wr=rook_from_shapefile("../examples/columbus.shp")
    >>> wr.pct_nonzero
    0.083298625572678045

    Notes
    -----

    Rook contiguity defines as neighbors any pair of polygons that share a
    common edge in their polygon definitions.

    See Also
    --------

    pysal.weights.W
    """
    return buildContiguity(shapefile,criteria='rook')

def bishop_from_shapefile(shapefile):
    """once set operations on W are implemented do something like

    q=queen_from_shapefile(shapefile)
    r=rook_from_shapefile(shapefile)
    return q-r

    """

    raise NotImplementedError


# Distance based weights

def knnW_from_array(array,k=2,p=2,ids=None):
    """
    Nearest neighbor weights from a numpy array

    Parameters
    ----------

    data       : array (n,k)
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
    pysal.weights.W
    """
    return knnW(array,k=k,p=p,ids=ids)

def knnW_from_shapefile(shapefile,k=2,p=2,ids=None):
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
    ids        : list
                 identifiers to attach to each observation
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
    >>> wc3=knnW_from_shapefile('../examples/columbus.shp',k=3)
    >>> wc3.weights[0]
    [1, 1, 1]
    >>> wc3.neighbors[0]
    [2, 1, 3]
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

    Supports polygon or point shapefiles

    Ties between neighbors of equal distance are arbitrarily broken.


    See Also
    --------
    pysal.weights.W
    """
    return knnW(shapefile,k=k,p=p,ids=ids)

if __name__ == "__main__":

    import doctest
    doctest.testmod()
