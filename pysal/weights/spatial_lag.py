"""
spatial lag operations

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

    Since it is general it does some checking to either determine if the y and
    w have been aligned.

    For use in a simulation context, the more optimized lag functions should
    be called directly since alignment of y and w can be assumed and the
    overhead of the alignment check thus avoided.

    Examples
    --------
    >>> from pysal.weights.weights import *
    >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
    >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
    >>> w=W({'weights':weights,'neighbors':neighbors})
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

    Or pass ids in on initial construction of W

    >>> w=W({'weights':weights,'neighbors':neighbors,'ids':['a','b','c']})
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
    >>> from pysal.weights.weights import *
    >>> w=lat2gal(3,3)
    >>> y=np.arange(9)
    >>> w.transform='r'
    >>> lag_array(w,y)
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
    >>> 

    """
    wy=np.zeros(y.shape,np.float)
    for i,id in enumerate(w.id_order):
        wy[i]=np.dot(w.weights[id],y[w.neighbors_0[id]])
    return wy 


def _timeing(order=20,iters=100,methods=(lag_array)):
    from pysal.weights.weights import lat2gal
    import time
    w=lat2gal(order,order)
    y=np.arange(order*order)
    for method in methods:
        t1=time.time()
        for i in range(iters):
            wy=method(w,y)
        t2=time.time()
        print method,t2-t1


if __name__ == '__main__':
    import doctest
    doctest.testmod()
