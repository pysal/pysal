"""
Distance based spatial weights

Author(s):
    Serge Rey srey@asu.edu
"""


import pysal
from pysal.common import *

def knnW(source,k=2,p=2,ids=None):
    """
    Creates contiguity matrix based on k nearest neighbors
    
    Parameters
    ----------

    source     : multitype
                 np.array  n observations on m attributes
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

    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wnn2=knnW(data,k=2)
    >>> wnn4=knnW(data,k=4)
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
    >>> wnn3e=knnW(data,p=2,k=3)
    >>> wnn3e.neighbors[0]
    [1, 5, 6]
    >>> wnn3m=knnW(data,p=1,k=3)
    >>> wnn3m.neighbors[0]
    [1, 5, 2]

    Point Shapefile

    >>> w=knnW('../examples/juvenile.shp')
    >>> w.pct_nonzero
    0.011904761904761904
    >>> w1=knnW('../examples/juvenile.shp',k=1)
    >>> w1.pct_nonzero
    0.0059523809523809521
    >>> 

    Notes
    -----

    Will be extended to support different source types.

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    pysal.weights.W
    """

    # handle source
    if type(source).__name__=='ndarray':
        data=source
    elif type(source)==type('string'):
        # assuming shapefile name
        sf=pysal.open(source)
        shapes=sf.read()
        if type(shapes[0]).__name__=='Polygon':
            data=np.array([shape.centroid for shape in shapes])
        elif type(shapes[0]).__name__=='Point':
            data=np.array([shape for shape in shapes])
        else:
            print 'Unsupported source type'
    else:
        print 'Unsupported source type'

    # calculate
    kd=KDTree(data)
    nnq=kd.query(data,k=k+1,p=p)
    info=nnq[1]
    neighbors={}
    weights={}
    for i,row in enumerate(info):
        neighbors[i]=row[1:].tolist()
        weights[i]=[1]*len(neighbors[i])

    return pysal.weights.W(neighbors,weights=weights,id_order=ids)

if __name__ == "__main__":

    import doctest
    doctest.testmod()
