"""
Set-like manipulation of weights matrices.
"""

__author__ = "Sergio J. Rey <srey@asu.edu>, Charles Schmidt <Charles.R.Schmidt@asu.edu>, David Folch <david.folch@asu.edu>"
__all__ = ['w_union', 'w_intersection', 'w_difference',
'w_symmetric_difference', 'w_subset']

import pysal
from pysal.common import copy


def w_union(w1, w2):
    """Returns a binary weights object, w, that includes all neighbor pairs that
    exist in either w1 or w2. 

    Parameters
    ----------

    w1      : W object

    w2      : W object


    Returns
    -------

    w       : W object


    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is 
    equivalent to the float ID 2.0. Returns a matrix with all the unique IDs
    from w1 and w2.


    Examples
    --------
    >>> import pysal
    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = w_union(w1, w2)
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14]
    >>> w2.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [19, 11, 14]
    >>>

    """
    neighbors = dict(w1.neighbors.items())
    for i in w2.neighbors:
        if i in neighbors:
            add_neigh = set(neighbors[i]).union(set(w2.neighbors[i]))
            neighbors[i] = list(add_neigh)
        else:
            neighbors[i] = copy.copy(w2.neighbors[i])
    return pysal.W(neighbors)


def w_intersection(w1, w2, w_shape='w1'):
    """Returns a binary weights object, w, that includes only those neighbor
    pairs that exist in both w1 and w2. 

    Parameters
    ----------

    w1      : W object

    w2      : W object

    w_shape : string
              Defines the shape of the returned weights matrix. 'w1' returns a
              matrix with the same IDs as w1; 'all' returns a matrix with all 
              the unique IDs from w1 and w2; and 'min' returns a matrix with 
              only the IDs occurring in both w1 and w2.


    Returns
    -------

    w       : W object


    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is 
    equivalent to the float ID 2.0. 


    Examples
    --------
    >>> import pysal
    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = w_intersection(w1, w2)
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14]
    >>> w2.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [11, 14]
    >>>

    """
    if w_shape == 'w1':
        neigh_keys = w1.neighbors.keys()
    elif w_shape == 'all':
        neigh_keys = set(w1.neighbors.keys()).union(set(w2.neighbors.keys()))
    elif w_shape == 'min':
        neigh_keys = set(w1.neighbors.keys()).intersection(set(w2.neighbors.keys()))
    else:
        raise Exception, "invalid string passed to w_shape"

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors and i in w2.neighbors:
            add_neigh = set(w1.neighbors[i]).intersection(set(w2.neighbors[i]))
            neighbors[i] = list(add_neigh)
        else:
            neighbors[i] = []

    return pysal.W(neighbors)


def w_difference(w1, w2, w_shape='w1', constrained=True):
    """Returns a binary weights object, w, that includes only neighbor pairs
    in w1 that are not in w2. The w_shape and constrained parameters
    determine which pairs in w1 that are not in w2 are returned.

    Parameters
    ----------

    w1      : W object

    w2      : W object

    w_shape : string
              Defines the shape of the returned weights matrix. 'w1' returns a
              matrix with the same IDs as w1; 'all' returns a matrix with all 
              the unique IDs from w1 and w2; and 'min' returns a matrix with
              the IDs occurring in w1 and not in w2.

    constrained : boolean
                  If False then the full set of neighbor pairs in w1 that are
                  not in w2 are returned. If True then those pairs that would 
                  not be possible if w_shape='min' are dropped. Ignored if 
                  w_shape is set to 'min'.


    Returns
    -------

    w       : W object


    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is 
    equivalent to the float ID 2.0. 


    Examples
    --------
    >>> import pysal
    >>> w1 = pysal.lat2W(4,4,rook=False)
    >>> w2 = pysal.lat2W(4,4,rook=True)
    >>> w = w_difference(w1, w2, constrained=False)
    >>> w1[0] == w[0]
    False
    >>> w1.neighbors[15]
    [10, 11, 14]
    >>> w2.neighbors[15]
    [11, 14]
    >>> w.neighbors[15]
    [10]
    >>>

    """
    if w_shape == 'w1':
        neigh_keys = w1.neighbors.keys()
    elif w_shape == 'all':
        neigh_keys = set(w1.neighbors.keys()).union(set(w2.neighbors.keys()))
    elif w_shape == 'min':
        neigh_keys = set(w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        if not neigh_keys:
            raise Exception, "returned an empty weights matrix"
    else:
        raise Exception, "invalid string passed to w_shape"

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors:
            if i in w2.neighbors:
                add_neigh = set(w1.neighbors[i]).difference(set(w2.neighbors[i]))
                neighbors[i] = list(add_neigh)
            else:
                neighbors[i] = copy.copy(w1.neighbors[i])
        else:
            neighbors[i] = []

    if constrained or w_shape == 'min':
        constrained_keys = set(w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        island_keys = set(neighbors.keys()).difference(constrained_keys)
        for i in island_keys:
            neighbors[i] = []
        for i in constrained_keys:
            neighbors[i] = list(set(neighbors[i]).intersection(constrained_keys))

    return pysal.W(neighbors)


def w_symmetric_difference(w1, w2, w_shape='all', constrained=True):
    """Returns a binary weights object, w, that includes only neighbor pairs
    that are not shared by w1 and w2. The w_shape and constrained parameters
    determine which pairs that are not shared by w1 and w2 are returned.

    Parameters
    ----------

    w1      : W object

    w2      : W object

    w_shape : string
              Defines the shape of the returned weights matrix. 'all' returns a
              matrix with all the unique IDs from w1 and w2; and 'min' returns 
              a matrix with the IDs not shared by w1 and w2.

    constrained : boolean
                  If False then the full set of neighbor pairs that are not
                  shared by w1 and w2 are returned. If True then those pairs 
                  that would not be possible if w_shape='min' are dropped. 
                  Ignored if w_shape is set to 'min'.


    Returns
    -------

    w       : W object


    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is 
    equivalent to the float ID 2.0. 


    Examples
    --------
    >>> import pysal
    >>> w1 = pysal.lat2W(4,4,rook=False)
    >>> w2 = pysal.lat2W(6,4,rook=True)
    >>> w = w_symmetric_difference(w1, w2, constrained=False)
    >>> w1[0] == w[0]
    False
    >>> w1.neighbors[15]
    [10, 11, 14]
    >>> w2.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [10, 19]
    >>>


    """
    if w_shape == 'all':
        neigh_keys = set(w1.neighbors.keys()).union(set(w2.neighbors.keys()))
    elif w_shape == 'min':
        neigh_keys = set(w1.neighbors.keys()).symmetric_difference(set(w2.neighbors.keys()))
    else:
        raise Exception, "invalid string passed to w_shape"

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors:
            if i in w2.neighbors:
                add_neigh = set(w1.neighbors[i]).symmetric_difference(set(w2.neighbors[i]))
                neighbors[i] = list(add_neigh)
            else:
                neighbors[i] = copy.copy(w1.neighbors[i])
        elif i in w2.neighbors:
            neighbors[i] = copy.copy(w2.neighbors[i])
        else:
            neighbors[i] = []

    if constrained or w_shape == 'min':
        constrained_keys = set(w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        island_keys = set(neighbors.keys()).difference(constrained_keys)
        for i in island_keys:
            neighbors[i] = []
        for i in constrained_keys:
            neighbors[i] = list(set(neighbors[i]).intersection(constrained_keys))

    return pysal.W(neighbors)


def w_subset(w1, ids):
    """Returns a binary weights object, w, that includes only those
    observations in ids.

    Parameters
    ----------

    w1      : W object

    ids     : list
              A list containing the IDs to be include in the returned weights
              object. 


    Returns
    -------

    w       : W object


    Examples
    --------
    >>> import pysal
    >>> w1 = pysal.lat2W(6,4)
    >>> ids = range(16)
    >>> w = w_subset(w1, ids)
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [11, 14]
    >>>

    """
    neighbors = {}
    ids = set(ids)
    for i in ids:
        if i in w1.neighbors:
            neigh_add = ids.intersection(set(w1.neighbors[i]))
            neighbors[i] = list(neigh_add)
        else:
            neighbors[i] = []

    return pysal.W(neighbors)

    
def _test():
    """"TEST"""
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()


