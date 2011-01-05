"""
spatial lag operations
"""
__authors__ = "Serge Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu>"
__all__ = ['lag_spatial']


def lag_spatial(w,y):
    """
    Spatial lag operator. If w is row standardized, returns the average of
    each observation's neighbors; if not, returns the weighted sum of each 
    observation's neighbors.

    Parameters
    ----------

    w : W
        weights object
    y : array
        variable to take the lag of (note: assumed that the order of y matches
        w.id_order)

    Returns
    -------

    wy : array
         array of numeric values for the spatial lag


    Examples
    --------
    >>> import pysal
    >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
    >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
    >>> id_order=['a','b','c']
    >>> w=pysal.W(neighbors,weights,id_order)
    >>> y = np.array([0,1,2])
    >>> lag_spatial(w,y)
    array([ 1.,  2.,  1.])
    >>> w.id_order=['b','c','a']
    >>> y = np.array([1,2,0])
    >>> lag_spatial(w,y)
    array([ 2.,  1.,  1.])
    >>> w=pysal.lat2W(3,3)
    >>> y=np.arange(9)
    >>> yl=lag_spatial(w,y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    >>> w.transform='r'
    >>> yl=lag_spatial(w,y)
    >>> yl
    array([ 2.        ,  2.        ,  3.        ,  3.33333333,  4.        ,
            4.66666667,  5.        ,  6.        ,  6.        ])
    >>> w.transform='b'
    >>> yl=lag_spatial(w,y)
    >>> yl
    array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
    >>> 

    """
    return w.sparse*y

if __name__ == '__main__':
    import doctest
    doctest.testmod()
