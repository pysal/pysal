"""
Spatial lag operations.
"""
__author__ = "Sergio J. Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu>, Levi John Wolf <ljw2@asu.edu"
__all__ = ['lag_spatial', 'lag_categorical']

import numpy as np
from pysal.common import iteritems as diter

def lag_spatial(w, y):
    """
    Spatial lag operator.

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------

    w                   : W
                          PySAL spatial weightsobject
    y                   : array
                          numpy array with dimensionality conforming to w (see examples)

    Returns
    -------

    wy                  : array
                          array of numeric values for the spatial lag

    Examples
    --------

    Setup a 9x9 binary spatial weights matrix and vector of data; compute the
    spatial lag of the vector.

    >>> import pysal
    >>> import numpy as np
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
    return w.sparse * y


def lag_categorical(w, y, ties='tryself'):
    """
    Spatial lag operator for categorical variables.

    Constructs the most common categories of neighboring observations, weighted
    by their weight strength.

    Parameters
    ----------

    w                   : W
                          PySAL spatial weightsobject
    y                   : iterable
                          iterable collection of categories (either int or
                          string) with dimensionality conforming to w (see examples)
    ties                : str
                          string describing the method to use when resolving
                          ties. By default, the option is "tryself",
                          and the category of the focal observation
                          is included with its neighbors to try
                          and break a tie. If this does not resolve the tie,
                          a winner is chosen randomly. To just use random choice to
                          break ties, pass "random" instead.
    Returns
    -------
    an (n x k) column vector containing the most common neighboring observation

    Notes
    -----
    This works on any array where the number of unique elements along the column
    axis is less than the number of elements in the array, for any dtype.
    That means the routine should work on any dtype that np.unique() can
    compare.

    Examples
    --------

    Set up a 9x9 weights matrix describing a 3x3 regular lattice. Lag one list of
    categorical variables with no ties.

    >>> import pysal
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> w = pysal.lat2W(3, 3)
    >>> y = ['a','b','a','b','c','b','c','b','c']
    >>> y_l = pysal.weights.spatial_lag.lag_categorical(w, y)
    >>> np.array_equal(y_l, np.array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']))
    True

    Explicitly reshape y into a (9x1) array and calculate lag again

    >>> yvect = np.array(y).reshape(9,1)
    >>> yvect_l = pysal.weights.spatial_lag.lag_categorical(w,yvect)
    >>> check = np.array( [ [i] for i in  ['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']] )
    >>> np.array_equal(yvect_l, check)
    True

    compute the lag of a 9x2 matrix of categories

    >>> y2 = ['a', 'c', 'c', 'd', 'b', 'a', 'd', 'd', 'c']
    >>> ym = np.vstack((y,y2)).T
    >>> ym_lag = pysal.weights.spatial_lag.lag_categorical(w,ym)
    >>> check = np.array([['b', 'b'], ['a', 'c'], ['b', 'c'], ['c', 'd'], ['b', 'd'], ['c', 'c'], ['b', 'd'], ['c', 'd'], ['b', 'b']])
    >>> np.array_equal(check, ym_lag)
    True

    """
    if isinstance(y, list):
        y = np.array(y)
    orig_shape = y.shape
    if len(orig_shape) > 1:
        if orig_shape[1] > 1:
            return np.vstack([lag_categorical(w,col) for col in y.T]).T
    y = y.flatten()
    output = np.zeros_like(y)
    keys = np.unique(y)
    inty = np.zeros(y.shape, dtype=np.int)
    for i,key in enumerate(keys):
       inty[y == key] = i
    for idx,neighbors in w:
        vals = np.zeros(keys.shape)
        for neighb, weight in diter(neighbors):
            vals[inty[w.id2i[neighb]]] += weight
        outidx = _resolve_ties(idx,inty,vals,neighbors,ties, w)
        output[w.id2i[idx]] = keys[outidx]
    return output.reshape(orig_shape)

def _resolve_ties(i,inty,vals,neighbors,method,w):
    """
    Helper function to resolve ties if lag is multimodal

    first, if this function gets called when there's actually no tie, then the
    correct value will be picked.

    if 'random' is selected as the method, a random tiebeaker is picked

    if 'tryself' is selected, then the observation's own value will be used in
    an attempt to break the tie, but if it fails, a random tiebreaker will be
    selected.
    """
    if len(vals[vals==vals.max()]) <= 1:
        return np.argmax(vals)
    elif method.lower() == 'random':
        ties = np.where(vals == vals.max())
        return np.random.choice(vals[ties])
    elif method.lower() == 'tryself':
        vals[inty[w.id2i[i]]] += np.mean(neighbors.values())
        return _resolve_ties(i,inty,vals,neighbors,'random', w)

