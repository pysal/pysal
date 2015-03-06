"""
Set-like manipulation of weights matrices.
"""

__author__ = "Sergio J. Rey <srey@asu.edu>, Charles Schmidt <schmidtc@gmail.com>, David Folch <david.folch@asu.edu>, Dani Arribas-Bel <darribas@asu.edu>"

import pysal
import copy
from scipy.sparse import isspmatrix_csr
from numpy import ones

__all__ = ['w_union', 'w_intersection', 'w_difference',
           'w_symmetric_difference', 'w_subset', 'w_clip']


def w_union(w1, w2, silent_island_warning=False):
    """
    Returns a binary weights object, w, that includes all neighbor pairs that
    exist in either w1 or w2.

    Parameters
    ----------

    w1                      : W 
                              object
    w2                      : W 
                              object
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------

    w       : W 
              object

    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is
    equivalent to the float ID 2.0. Returns a matrix with all the unique IDs
    from w1 and w2.

    Examples
    --------

    Construct rook weights matrices for two regions, one is 4x4 (16 areas)
    and the other is 6x4 (24 areas). A union of these two weights matrices
    results in the new weights matrix matching the larger one.

    >>> import pysal
    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = pysal.weights.w_union(w1, w2)
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
    return pysal.W(neighbors, silent_island_warning=silent_island_warning)


def w_intersection(w1, w2, w_shape='w1', silent_island_warning=False):
    """
    Returns a binary weights object, w, that includes only 
    those neighbor pairs that exist in both w1 and w2.

    Parameters
    ----------

    w1                      : W 
                              object
    w2                      : W 
                              object
    w_shape                 : string
                              Defines the shape of the returned weights matrix. 'w1' returns a
                              matrix with the same IDs as w1; 'all' returns a matrix with all
                              the unique IDs from w1 and w2; and 'min' returns a matrix with
                              only the IDs occurring in both w1 and w2.
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------

    w       : W 
              object

    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is
    equivalent to the float ID 2.0.

    Examples
    --------

    Construct rook weights matrices for two regions, one is 4x4 (16 areas)
    and the other is 6x4 (24 areas). An intersection of these two weights
    matrices results in the new weights matrix matching the smaller one.

    >>> import pysal
    >>> w1 = pysal.lat2W(4,4)
    >>> w2 = pysal.lat2W(6,4)
    >>> w = pysal.weights.w_intersection(w1, w2)
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
        neigh_keys = set(w1.neighbors.keys(
        )).intersection(set(w2.neighbors.keys()))
    else:
        raise Exception("invalid string passed to w_shape")

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors and i in w2.neighbors:
            add_neigh = set(w1.neighbors[i]).intersection(set(w2.neighbors[i]))
            neighbors[i] = list(add_neigh)
        else:
            neighbors[i] = []

    return pysal.W(neighbors, silent_island_warning=silent_island_warning)


def w_difference(w1, w2, w_shape='w1', constrained=True, silent_island_warning=False):
    """
    Returns a binary weights object, w, that includes only neighbor pairs
    in w1 that are not in w2. The w_shape and constrained parameters
    determine which pairs in w1 that are not in w2 are returned.

    Parameters
    ----------

    w1                      : W 
                              object
    w2                      : W 
                              object
    w_shape                 : string
                              Defines the shape of the returned weights matrix. 'w1' returns a
                              matrix with the same IDs as w1; 'all' returns a matrix with all
                              the unique IDs from w1 and w2; and 'min' returns a matrix with
                              the IDs occurring in w1 and not in w2.
    constrained             : boolean
                              If False then the full set of neighbor pairs in w1 that are
                              not in w2 are returned. If True then those pairs that would
                              not be possible if w_shape='min' are dropped. Ignored if
                              w_shape is set to 'min'.
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------

    w       : W 
              object

    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is
    equivalent to the float ID 2.0.

    Examples
    --------

    Construct rook (w2) and queen (w1) weights matrices for two 4x4 regions
    (16 areas). A queen matrix has all the joins a rook matrix does plus joins
    between areas that share a corner. The new matrix formed by the difference
    of rook from queen contains only join at corners (typically called a
    bishop matrix). Note that the difference of queen from rook would result
    in a weights matrix with no joins.

    >>> import pysal
    >>> w1 = pysal.lat2W(4,4,rook=False)
    >>> w2 = pysal.lat2W(4,4,rook=True)
    >>> w = pysal.weights.w_difference(w1, w2, constrained=False)
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
        neigh_keys = set(
            w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        if not neigh_keys:
            raise Exception("returned an empty weights matrix")
    else:
        raise Exception("invalid string passed to w_shape")

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors:
            if i in w2.neighbors:
                add_neigh = set(w1.neighbors[i]
                                ).difference(set(w2.neighbors[i]))
                neighbors[i] = list(add_neigh)
            else:
                neighbors[i] = copy.copy(w1.neighbors[i])
        else:
            neighbors[i] = []

    if constrained or w_shape == 'min':
        constrained_keys = set(
            w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        island_keys = set(neighbors.keys()).difference(constrained_keys)
        for i in island_keys:
            neighbors[i] = []
        for i in constrained_keys:
            neighbors[i] = list(
                set(neighbors[i]).intersection(constrained_keys))

    return pysal.W(neighbors, silent_island_warning=silent_island_warning)


def w_symmetric_difference(w1, w2, w_shape='all', constrained=True, silent_island_warning=False):
    """
    Returns a binary weights object, w, that includes only neighbor pairs
    that are not shared by w1 and w2. The w_shape and constrained parameters
    determine which pairs that are not shared by w1 and w2 are returned.

    Parameters
    ----------

    w1                      : W 
                              object
    w2                      : W 
                              object
    w_shape                 : string
                              Defines the shape of the returned weights matrix. 'all' returns a
                              matrix with all the unique IDs from w1 and w2; and 'min' returns
                              a matrix with the IDs not shared by w1 and w2.
    constrained             : boolean
                              If False then the full set of neighbor pairs that are not
                              shared by w1 and w2 are returned. If True then those pairs
                              that would not be possible if w_shape='min' are dropped.
                              Ignored if w_shape is set to 'min'.
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------

    w       : W 
              object

    Notes
    -----
    ID comparisons are performed using ==, therefore the integer ID 2 is
    equivalent to the float ID 2.0.

    Examples
    --------

    Construct queen weights matrix for a 4x4 (16 areas) region (w1) and a rook
    matrix for a 6x4 (24 areas) region (w2). The symmetric difference of these
    two matrices (with w_shape set to 'all' and constrained set to False)
    contains the corner joins in the overlap area, all the joins in the
    non-overlap area.

    >>> import pysal
    >>> w1 = pysal.lat2W(4,4,rook=False)
    >>> w2 = pysal.lat2W(6,4,rook=True)
    >>> w = pysal.weights.w_symmetric_difference(w1, w2, constrained=False)
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
        neigh_keys = set(w1.neighbors.keys(
        )).symmetric_difference(set(w2.neighbors.keys()))
    else:
        raise Exception("invalid string passed to w_shape")

    neighbors = {}
    for i in neigh_keys:
        if i in w1.neighbors:
            if i in w2.neighbors:
                add_neigh = set(w1.neighbors[i]).symmetric_difference(
                    set(w2.neighbors[i]))
                neighbors[i] = list(add_neigh)
            else:
                neighbors[i] = copy.copy(w1.neighbors[i])
        elif i in w2.neighbors:
            neighbors[i] = copy.copy(w2.neighbors[i])
        else:
            neighbors[i] = []

    if constrained or w_shape == 'min':
        constrained_keys = set(
            w1.neighbors.keys()).difference(set(w2.neighbors.keys()))
        island_keys = set(neighbors.keys()).difference(constrained_keys)
        for i in island_keys:
            neighbors[i] = []
        for i in constrained_keys:
            neighbors[i] = list(
                set(neighbors[i]).intersection(constrained_keys))

    return pysal.W(neighbors, silent_island_warning=silent_island_warning)


def w_subset(w1, ids, silent_island_warning=False):
    """
    Returns a binary weights object, w, that includes only those
    observations in ids.

    Parameters
    ----------

    w1                      : W 
                              object
    ids                     : list
                              A list containing the IDs to be include in the returned weights
                              object.
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------

    w       : W 
              object

    Examples
    --------

    Construct a rook weights matrix for a 6x4 region (24 areas). By default
    PySAL assigns integer IDs to the areas in a region. By passing in a list
    of integers from 0 to 15, the first 16 areas are extracted from the
    previous weights matrix, and only those joins relevant to the new region
    are retained.

    >>> import pysal
    >>> w1 = pysal.lat2W(6,4)
    >>> ids = range(16)
    >>> w = pysal.weights.w_subset(w1, ids)
    >>> w1[0] == w[0]
    True
    >>> w1.neighbors[15]
    [11, 14, 19]
    >>> w.neighbors[15]
    [11, 14]
    >>>

    """ 

    neighbors = {}
    ids_set = set(ids)
    for i in ids:
        if i in w1.neighbors:
            neigh_add = ids_set.intersection(set(w1.neighbors[i]))
            neighbors[i] = list(neigh_add)
        else:
            neighbors[i] = []

    return pysal.W(neighbors, id_order=ids, silent_island_warning=silent_island_warning)


def w_clip(w1, w2, outSP=True, silent_island_warning=False):
    '''
    Clip a continuous W object (w1) with a different W object (w2) so only cells where
    w2 has a non-zero value remain with non-zero values in w1.

    Checks on w1 and w2 are performed to make sure they conform to the
    appropriate format and, if not, they are converted.

    Parameters
    ----------
    w1                      : W
                              pysal.W, scipy.sparse.csr.csr_matrix
                              Potentially continuous weights matrix to be clipped. The clipped
                              matrix wc will have at most the same elements as w1.
    w2                      : W
                              pysal.W, scipy.sparse.csr.csr_matrix
                              Weights matrix to use as shell to clip w1. Automatically
                              converted to binary format. Only non-zero elements in w2 will be
                              kept non-zero in wc. NOTE: assumed to be of the same shape as w1
    outSP                   : boolean
                              If True (default) return sparse version of the clipped W, if
                              False, return pysal.W object of the clipped matrix
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------
    wc      : W
              pysal.W, scipy.sparse.csr.csr_matrix
              Clipped W object (sparse if outSP=Ture). It inherits
              ``id_order`` from w1.

    Examples
    --------
    >>> import pysal as ps

    First create a W object from a lattice using queen contiguity and
    row-standardize it (note that these weights will stay when we clip the
    object, but they will not neccesarily represent a row-standardization
    anymore):

    >>> w1 = ps.lat2W(3, 2, rook=False)
    >>> w1.transform = 'R'

    We will clip that geography assuming observations 0, 2, 3 and 4 belong to
    one group and 1, 5 belong to another group and we don't want both groups
    to interact with each other in our weights (i.e. w_ij = 0 if i and j in
    different groups). For that, we use the following method:

    >>> w2 = ps.block_weights(['r1', 'r2', 'r1', 'r1', 'r1', 'r2'])

    To illustrate that w2 will only be considered as binary even when the
    object passed is not, we can row-standardize it

    >>> w2.transform = 'R'

    The clipped object ``wc`` will contain only the spatial queen
    relationships that occur within one group ('r1' or 'r2') but will have
    gotten rid of those that happen across groups

    >>> wcs = ps.weights.Wsets.w_clip(w1, w2, outSP=True)

    This will create a sparse object (recommended when n is large).

    >>> wcs.sparse.toarray()
    array([[ 0.        ,  0.        ,  0.33333333,  0.33333333,  0.        ,
             0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ],
           [ 0.2       ,  0.        ,  0.        ,  0.2       ,  0.2       ,
             0.        ],
           [ 0.2       ,  0.        ,  0.2       ,  0.        ,  0.2       ,
             0.        ],
           [ 0.        ,  0.        ,  0.33333333,  0.33333333,  0.        ,
             0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ]])

    If we wanted an original W object, we can control that with the argument
    ``outSP``:

    >>> wc = ps.weights.Wsets.w_clip(w1, w2, outSP=False)
    WARNING: there are 2 disconnected observations
    Island ids:  [1, 5]
    >>> wc.full()[0]
    array([[ 0.        ,  0.        ,  0.33333333,  0.33333333,  0.        ,
             0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ],
           [ 0.2       ,  0.        ,  0.        ,  0.2       ,  0.2       ,
             0.        ],
           [ 0.2       ,  0.        ,  0.2       ,  0.        ,  0.2       ,
             0.        ],
           [ 0.        ,  0.        ,  0.33333333,  0.33333333,  0.        ,
             0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ]])

    You can check they are actually the same:

    >>> wcs.sparse.toarray() == wc.full()[0]
    array([[ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True]], dtype=bool)

    '''
    
    if not w1.id_order:
        w1.id_order = None
    id_order = w1.id_order
    if not isspmatrix_csr(w1):
        w1 = w1.sparse
    if not isspmatrix_csr(w2):
        w2 = w2.sparse
    w2.data = ones(w2.data.shape)
    wc = w1.multiply(w2)
    wc = pysal.weights.WSP(wc, id_order=id_order)
    if not outSP:
        wc = pysal.weights.WSP2W(wc, silent_island_warning=silent_island_warning)
    return wc


