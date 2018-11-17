"""
Spatial lag operations.
"""
__author__ = "Sergio J. Rey <srey@asu.edu>, David C. Folch <david.folch@asu.edu>, Levi John Wolf <ljw2@asu.edu"
__all__ = ['lag_spatial', 'lag_categorical']

import numpy as np

def lag_spatial(w, y):
    """
    Spatial lag operator.

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------

    w                   : W
                          libpysal spatial weightsobject
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

    >>> import libpysal
    >>> import numpy as np
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = np.arange(9)
    >>> yl = libpysal.weights.spatial_lag.lag_spatial(w, y)
    >>> yl
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardize the weights matrix and recompute the spatial lag

    >>> w.transform = 'r'
    >>> yl = libpysal.weights.spatial_lag.lag_spatial(w, y)
    >>> yl
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])


    Explicitly define data vector as 9x1 and recompute the spatial lag

    >>> y.shape = (9, 1)
    >>> yl = libpysal.weights.spatial_lag.lag_spatial(w, y)
    >>> yl
    array([[2.        ],
           [2.        ],
           [3.        ],
           [3.33333333],
           [4.        ],
           [4.66666667],
           [5.        ],
           [6.        ],
           [6.        ]])


    Take the spatial lag of a 9x2 data matrix

    >>> yr = np.arange(8, -1, -1)
    >>> yr.shape = (9, 1)
    >>> x = np.hstack((y, yr))
    >>> yl = libpysal.weights.spatial_lag.lag_spatial(w, x)
    >>> yl
    array([[2.        , 6.        ],
           [2.        , 6.        ],
           [3.        , 5.        ],
           [3.33333333, 4.66666667],
           [4.        , 4.        ],
           [4.66666667, 3.33333333],
           [5.        , 3.        ],
           [6.        , 2.        ],
           [6.        , 2.        ]])

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

    >>> import libpysal
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = ['a','b','a','b','c','b','c','b','c']
    >>> y_l = libpysal.weights.spatial_lag.lag_categorical(w, y)
    >>> np.array_equal(y_l, np.array(['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']))
    True

    Explicitly reshape y into a (9x1) array and calculate lag again

    >>> yvect = np.array(y).reshape(9,1)
    >>> yvect_l = libpysal.weights.spatial_lag.lag_categorical(w,yvect)
    >>> check = np.array( [ [i] for i in  ['b', 'a', 'b', 'c', 'b', 'c', 'b', 'c', 'b']] )
    >>> np.array_equal(yvect_l, check)
    True

    compute the lag of a 9x2 matrix of categories

    >>> y2 = ['a', 'c', 'c', 'd', 'b', 'a', 'd', 'd', 'c']
    >>> ym = np.vstack((y,y2)).T
    >>> ym_lag = libpysal.weights.spatial_lag.lag_categorical(w,ym)
    >>> check = np.array([['b', 'd'], ['a', 'c'], ['b', 'c'], ['c', 'd'], ['b', 'd'], ['c', 'c'], ['b', 'd'], ['c', 'd'], ['b', 'c']])
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
    labels = np.unique(y)
    normalized_labels = np.zeros(y.shape, dtype=np.int)
    for i,label in enumerate(labels):
       normalized_labels[y == label] = i
    for focal_name,neighbors in w:
        focal_idx = w.id2i[focal_name]
        neighborhood_tally = np.zeros(labels.shape)
        for neighb_name, weight in list(neighbors.items()):
            neighb_idx = w.id2i[neighb_name]
            neighb_label = normalized_labels[neighb_idx]
            neighborhood_tally[neighb_label] += weight
        out_label_idx = _resolve_ties(focal_idx, normalized_labels,
                               neighborhood_tally, neighbors, ties, w)
        output[focal_idx] = labels[out_label_idx]
    return output.reshape(orig_shape)


def _resolve_ties(idx,normalized_labels,tally,neighbors,method,w):
    """
    Helper function to resolve ties if lag is multimodal

    first, if this function gets called when there's actually no tie, then the
    correct value will be picked.

    if 'random' is selected as the method, a random tiebeaker is picked

    if 'tryself' is selected, then the observation's own value will be used in
    an attempt to break the tie, but if it fails, a random tiebreaker will be
    selected.

    Arguments
    ---------
    idx                 : int
                          index (aligned with `normalized_labels`) of the 
                          current observation being resolved.
    normalized_labels   : (n,) array of ints
                          normalized array of labels for each observation
    tally               : (p,) array of floats
                          current tally of neighbors' labels around `idx` to resolve.
    neighbors           : dict of (neighbor_name : weight)
                          the elements of the weights object, identical to w[idx]
    method              : string
                          configuration option to use a specific tiebreaking method. 
                          supported options are:
                          1. tryself: Use the focal observation's label to tiebreak.
                                      If this doesn't successfully break the tie, 
                                      (which only occurs if it induces a new tie),
                                      decide randomly. 
                          2. random: Resolve the tie randomly amongst winners.
                          3. lowest: Pick the lowest-value label amongst winners.
                          4. highest: Pick the highest-value label amongst winners.
    w                   : pysal.W object
                          a PySAL weights object aligned with normalized_labels. 

    Returns
    -------
    integer denoting which label to use to label the observation.
    """
    ties, = np.where(tally == tally.max()) #returns a tuple for flat arrays
    if len(tally[tally==tally.max()]) <= 1: #no tie, pick the highest
        return np.argmax(tally).astype(int)
    elif method.lower() == 'random': #choose randomly from tally
        return np.random.choice(np.squeeze(ties)).astype(int)
    elif method.lower() == 'lowest': # pick lowest tied value
        return ties[0].astype(int)
    elif method.lower() == 'highest': #pick highest tied value
        return ties[-1].astype(int)
    elif method.lower() == 'tryself': # add self-label as observation, try again, random if fail
        mean_neighbor_value = np.mean(list(neighbors.values()))
        tally[normalized_labels[idx]] += mean_neighbor_value
        return _resolve_ties(idx,normalized_labels,tally,neighbors,'random', w)
    else:
        raise KeyError('Tie-breaking method for categorical lag not recognized')
