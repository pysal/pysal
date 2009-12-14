"""
spatial lag operations

"""

from pysal.common import *


def lag(y,w):
    """General spatial lag operator


    Parameters
    ----------

    y : list of numeric values

    w : weights object

    Returns
    -------

    yl : np.ndarray of numeric values for the spatial lag

    Since it is general it does some checking to either determine if the y and
    w have been aligned.

    For use in a simulation context, the more optimized lag functions should
    be called directly since alignment of y and w can be assumed and the
    overhead of the alignment check thus avoided.

    Example
    -------
    >>> from pysal.weights.weights import *
    >>> w=lat2gal(3,3)
    >>> y=np.arange(9)
    >>> lag(y,w)
    Traceback (most recent call last):
      File "/Library/Frameworks/Python.framework/Versions/4.3.0/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest __main__.lag[3]>", line 1, in <module>
        lag(y,w)
      File "spatial_lag.py", line 56, in lag
        raise ValueError("w id_order must be set to align with y's order")
    ValueError: w id_order must be set to align with y's order
    >>> w.id_order=range(9)
    >>> yl=lag(y,w)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    """

    if not w.id_order_set: 
        raise ValueError("w id_order must be set to align with y's order")
    else:
        return lag_dict(y,w)

def lag_dict(y,w):
    """Optimized spatial lag operator

    Assumes y and w are already aligned and omits any sanity checks

    Parameters
    ----------

    y : list of numeric values

    w : weights object

    Returns
    -------

    yl : np.ndarray of numeric values for the spatial lag


    Example
    -------
    >>> from pysal.weights.weights import *
    >>> w=lat2gal(3,3)
    >>> y=np.arange(9)
    >>> yl=lag_dict(y,w)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])

    Notes
    -----
    old unoptimized code is as follows for reference::

        def _lag_dict(y,w):
            yl=np.zeros(y.shape,'float')
            for i,wi in enumerate(w):
                for j,wij in wi.items():
                    yl[i]+=wij*y[w.id_order.index(j)]
            return yl
"""
    yl=[sum([wij*y[w.id_order.index(j)] for j,wij in wi.items()]) for wi in w]
    return np.array(yl)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
