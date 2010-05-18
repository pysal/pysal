"""
spatial lag operations


we need to dump the indexing scheme and have the user set y's order before
calling lag. too expensive.

"""

from pysal.common import *


def lag(w,y):
    """General spatial lag operator

    Parameters
    ----------
    w : W
        weights object
    y : array
        variable to take the lag of

    Returns
    -------
    wy : np.ndarray
         values for the spatial lag

    Notes
    -----

    Because the weights and attribute data may be from different sources
    we have to ensure that the ordering of y and w has been aligned.

    For use in a simulation context, the more optimized lag functions should
    be called directly since alignment of y and w can be assumed and the
    overhead of the alignment check thus avoided.

    Examples
    --------
    >>> from pysal.weights import *
    >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
    >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
    >>> w=W(neighbors,weights)
    >>> y=np.arange(3)

    y and W are not yet aligned

    >>> lag(w,y)
    Traceback (most recent call last):
      File "<ipython console>", line 1, in <module>
      File "/Users/serge/Research/p/Pysal/src/google/trunk/pysal/esda/spatial_lag.py", line 43, in lag
        raise ValueError("w id_order must be set to align with y's order")
    ValueError: w id_order must be set to align with y's order

    To align y and w, use the id_order property

    >>> w.id_order=['a','b','c']
    >>> lag(w,y)
    array([ 1.,  2.,  1.])

    The alingment can be arbitrary

    >>> w.id_order=['b','c','a']
    >>> y = np.array([1,2,0])
    >>> lag(w,y)
    array([ 2.,  1.,  1.])


    Or pass ids in on initial construction of W

    >>> w.id_order=['a','b','c']
    >>> y = np.array([0,1,2])
    >>> w=W(neighbors,weights,['a','b','c'])
    >>> lag(w,y)
    array([ 1.,  2.,  1.])
    """

    if not w.id_order_set: 
        raise ValueError("w id_order must be set to align with y's order")
    else:
        return lag_array(w,y)

def lag_array(w,y):
    """Spatial lag operator using np.arrays

    Assumes y and w are already aligned and omits any sanity checks

    Parameters
    ----------

    w : W
        weights object
    y : array
        variable to take the lag of

    Returns
    -------

    wy : np.ndarray of numeric values for the spatial lag


    Examples
    --------
    >>> from pysal.weights import *
    >>> w=lat2W(3,3)
    >>> y=np.arange(9)
    >>> w.transform='r'
    >>> lag_array(w,y)
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
    >>> 

    """
    wy=np.zeros(y.shape,np.float)
    for i,id in enumerate(w.id_order):
        wy[i]=np.dot(w.weights[id],y[w.neighbor_offsets[id]])
    return wy 

def lag_sparse(w,y):
    """
    Spatial lag using sparse attribute of W

    Parameters
    ----------

    w : W
        weights object
    y : array
        variable to take the lag of

    Returns
    -------

    wy : np.ndarray of numeric values for the spatial lag


    Examples
    --------
    >>> import pysal
    >>> w=pysal.lat2W(3,3)
    >>> y=np.arange(9)
    >>> yl=lag_sparse(w,y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    >>> w.transform='r'
    >>> yl=lag_sparse(w,y)
    >>> yl
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
    >>> w.transform='b'
    >>> yl=lag_sparse(w,y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    >>> 


    Notes
    -----

    Speed ups will be machine specific so we dont run the doctest here, but
    the following should give you an idea of the relative speed boosts:

    import time
    w=pysal.lat2W(100,100)
    y=np.arange(100**2)
    import spatial_lag
    import time
    t1=time.time();spatial_lag.lag_array(w,y);time.time()-t1
    array([   101.,    103.,    106., ...,  29891.,  29894.,  19897.])
    0.31401681900024414
    t1=time.time();spatial_lag.lag_sparse(w,y);time.time()-t1
    array([   101.,    103.,    106., ...,  29891.,  29894.,  19897.])
    0.0014200210571289062
    
    """

    return w.sparse*y

if __name__ == '__main__':
    import doctest
    doctest.testmod()
