"""
spatial lag operations
"""
__authors__ = "Serge Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu>"
__all__ = ['lag_spatial']

def lag_spatial(w, y):
    """
    Spatial lag operator. If w is row standardized, returns the average of
    each observation's neighbors; if not, returns the weighted sum of each 
    observation's neighbors.

    Parameters
    ----------

    w : W
        weights object
    y : array
        numpy array with dimensionality conforming to w (see examples)

    Returns
    -------

    wy : array
         array of numeric values for the spatial lag


    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    
    Setup a 9x9 binary spatial weights matrix and vector of data; compute the
    spatial lag of the vector.

    >>> w = pysal.lat2W(3, 3)
    >>> y = np.arange(9)
    >>> yl = pysal.lag_spatial(w, y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    
    Row standardize the weights matrix and recompute the spatial lag

    >>> w.transform = 'r'
    >>> yl = pysal.lag_spatial(w, y)
    >>> yl
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
    
    Explicitly define data vector as 9x1 and recompute the spatial lag

    >>> y.shape = (9, 1)
    >>> yl = pysal.lag_spatial(w, y)
    >>> yl
    array([[ 2.        ],
           [ 2.        ],
           [ 3.        ],
           [ 3.33333333],
           [ 4.        ],
           [ 4.66666667],
           [ 5.        ],
           [ 6.        ],
           [ 6.        ]])

    Take the spatial lag of a 9x2 data matrix

    >>> yr = np.arange(8, -1, -1)
    >>> yr.shape = (9, 1)
    >>> x = np.hstack((y, yr))
    >>> yl = pysal.lag_spatial(w, x)
    >>> yl
    array([[ 2.        ,  6.        ],
           [ 2.        ,  6.        ],
           [ 3.        ,  5.        ],
           [ 3.33333333,  4.66666667],
           [ 4.        ,  4.        ],
           [ 4.66666667,  3.33333333],
           [ 5.        ,  3.        ],
           [ 6.        ,  2.        ],
           [ 6.        ,  2.        ]])

    """
    return w.sparse*y

def _test():
    """Doc test"""
    import doctest
    doctest.testmod(verbose=True)


if __name__ == '__main__':
    _test()
