from ..io.fileio import FileIO as psopen
from .weights import W, WSP
from .set_operations import w_subset
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
import copy
import scipy.spatial
import os
import scipy
from warnings import warn
import numbers
from collections import defaultdict
from ..common import requires

try:
    import geopandas as gpd
except ImportError:
    warn('geopandas not available. Some functionality will be disabled.')

__all__ = ['lat2W', 'block_weights', 'comb', 'order', 'higher_order',
           'shimbel', 'remap_ids', 'full2W', 'full', 'WSP2W',
           'insert_diagonal', 'get_ids', 'get_points_array_from_shapefile',
           'min_threshold_distance', 'lat2SW', 'w_local_cluster',
           'higher_order_sp', 'hexLat2W', 'attach_islands',
           'nonplanar_neighbors', 'fuzzy_contiguity']


KDTREE_TYPES = [scipy.spatial.KDTree, scipy.spatial.cKDTree]

def hexLat2W(nrows=5, ncols=5):
    """
    Create a W object for a hexagonal lattice.

    Parameters
    ----------

    nrows   : int
              number of rows
    ncols   : int
              number of columns

    Returns
    -------

    w : W
        instance of spatial weights class W

    Notes
    -----

    Observations are row ordered: first k observations are in row 0, next k in row 1, and so on.

    Construction is based on shifting every other column of a regular lattice
    down 1/2 of a cell.

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> w = lat2W()
    >>> w.neighbors[1]
    [0, 6, 2]
    >>> w.neighbors[21]
    [16, 20, 22]
    >>> wh = hexLat2W()
    >>> wh.neighbors[1]
    [0, 6, 2, 5, 7]
    >>> wh.neighbors[21]
    [16, 20, 22]
    >>>
    """

    if nrows == 1 or ncols == 1:
        print("Hexagon lattice requires at least 2 rows and columns")
        print("Returning a linear contiguity structure")
        return lat2W(nrows, ncols)

    n = nrows * ncols
    rid = [i // ncols for i in range(n)]
    cid = [i % ncols for i in range(n)]
    r1 = nrows - 1
    c1 = ncols - 1

    w = lat2W(nrows, ncols).neighbors
    for i in range(n):
        odd = cid[i] % 2
        if odd:
            if rid[i] < r1:  # odd col index above last row
                # new sw neighbor
                if cid[i] > 0:
                    j = i + ncols - 1
                    w[i] = w.get(i, []) + [j]
                # new se neighbor
                if cid[i] < c1:
                    j = i + ncols + 1
                    w[i] = w.get(i, []) + [j]

        else:  # even col
            # nw
            jnw = [i - ncols - 1]
            # ne
            jne = [i - ncols + 1]
            if rid[i] > 0:
                w[i]
                if cid[i] == 0:
                    w[i] = w.get(i, []) + jne
                elif cid[i] == c1:
                    w[i] = w.get(i, []) + jnw
                else:
                    w[i] = w.get(i, []) + jne
                    w[i] = w.get(i, []) + jnw


    return W(w)


def lat2W(nrows=5, ncols=5, rook=True, id_type='int'):
    """
    Create a W object for a regular lattice.

    Parameters
    ----------

    nrows   : int
              number of rows
    ncols   : int
              number of columns
    rook    : boolean
              type of contiguity. Default is rook. For queen, rook =False
    id_type : string
              string defining the type of IDs to use in the final W object;
              options are 'int' (0, 1, 2 ...; default), 'float' (0.0,
              1.0, 2.0, ...) and 'string' ('id0', 'id1', 'id2', ...)

    Returns
    -------

    w : W
        instance of spatial weights class W

    Notes
    -----

    Observations are row ordered: first k observations are in row 0, next k in row 1, and so on.

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> w9 = lat2W(3,3)
    >>> "%.3f"%w9.pct_nonzero
    '29.630'
    >>> w9[0] == {1: 1.0, 3: 1.0}
    True
    >>> w9[3] == {0: 1.0, 4: 1.0, 6: 1.0}
    True
    """
    n = nrows * ncols
    r1 = nrows - 1
    c1 = ncols - 1
    rid = [i // ncols for i in range(n)] #must be floor!
    cid = [i % ncols for i in range(n)]
    w = {}
    r = below = 0
    for i in range(n - 1):
        if rid[i] < r1:
            below = rid[i] + 1
            r = below * ncols + cid[i]
            w[i] = w.get(i, []) + [r]
            w[r] = w.get(r, []) + [i]
        if cid[i] < c1:
            right = cid[i] + 1
            c = rid[i] * ncols + right
            w[i] = w.get(i, []) + [c]
            w[c] = w.get(c, []) + [i]
        if not rook:
            # southeast bishop
            if cid[i] < c1 and rid[i] < r1:
                r = (rid[i] + 1) * ncols + 1 + cid[i]
                w[i] = w.get(i, []) + [r]
                w[r] = w.get(r, []) + [i]
            # southwest bishop
            if cid[i] > 0 and rid[i] < r1:
                r = (rid[i] + 1) * ncols - 1 + cid[i]
                w[i] = w.get(i, []) + [r]
                w[r] = w.get(r, []) + [i]

    neighbors = {}
    weights = {}
    for key in w:
        weights[key] = [1.] * len(w[key])
    ids = list(range(n))
    if id_type == 'string':
        ids = ['id' + str(i) for i in ids]
    elif id_type == 'float':
        ids = [i * 1. for i in ids]
    if id_type == 'string' or id_type == 'float':
        id_dict = dict(list(zip(list(range(n)), ids)))
        alt_w = {}
        alt_weights = {}
        for i in w:
            values = [id_dict[j] for j in w[i]]
            key = id_dict[i]
            alt_w[key] = values
            alt_weights[key] = weights[i]
        w = alt_w
        weights = alt_weights
    return W(w, weights, ids=ids, id_order=ids[:])


def block_weights(regimes, ids=None, sparse=False):
    """
    Construct spatial weights for regime neighbors.

    Block contiguity structures are relevant when defining neighbor relations
    based on membership in a regime. For example, all counties belonging to
    the same state could be defined as neighbors, in an analysis of all
    counties in the US.

    Parameters
    ----------
    regimes     : list, array
                  ids of which regime an observation belongs to
    ids         : list, array
                  Ordered sequence of IDs for the observations
    sparse      : boolean
                  If True return WSP instance
                  If False return W instance

    Returns
    -------

    W : spatial weights instance

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> import numpy as np
    >>> regimes = np.ones(25)
    >>> regimes[range(10,20)] = 2
    >>> regimes[range(21,25)] = 3
    >>> regimes
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2.,
           2., 2., 2., 1., 3., 3., 3., 3.])
    >>> w = block_weights(regimes)
    >>> w.weights[0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> w.neighbors[0]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
    >>> regimes = ['n','n','s','s','e','e','w','w','e']
    >>> n = len(regimes)
    >>> w = block_weights(regimes)
    >>> w.neighbors == {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}
    True
    """
    rids = np.unique(regimes)
    neighbors = {}
    NPNZ = np.nonzero
    regimes = np.array(regimes)
    for rid in rids:
        members = NPNZ(regimes == rid)[0]
        for member in members:
            neighbors[member] = members[NPNZ(members != member)[0]].tolist()
    w = W(neighbors)
    if ids is not None:
        w.remap_ids(ids)
    if sparse:
        w = WSP(w.sparse, id_order=ids)
    return w


def comb(items, n=None):
    """
    Combinations of size n taken from items

    Parameters
    ----------

    items : list
            items to be drawn from
    n     : integer
            size of combinations to take from items

    Returns
    -------

    implicit : generator
               combinations of size n taken from items

    Examples
    --------
    >>> x = range(4)
    >>> for c in comb(x, 2):
    ...     print(c)
    ...
    [0, 1]
    [0, 2]
    [0, 3]
    [1, 2]
    [1, 3]
    [2, 3]

    """
    items = list(items)
    if n is None:
        n = len(items)
    for i in list(range(len(items))):
        v = items[i:i + 1]
        if n == 1:
            yield v
        else:
            rest = items[i + 1:]
            for c in comb(rest, n - 1):
                yield v + c


def order(w, kmax=3):
    """
    Determine the non-redundant order of contiguity up to a specific
    order.

    Parameters
    ----------

    w       : W
              spatial weights object

    kmax    : int
              maximum order of contiguity

    Returns
    -------

    info : dictionary
           observation id is the key, value is a list of contiguity
           orders with a negative 1 in the ith position

    Notes
    -----
    Implements the algorithm in Anselin and Smirnov (1996) [Anselin1996b]_

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> from libpysal.weights.contiguity import Rook
    >>> import libpysal
    >>> w = Rook.from_shapefile(libpysal.examples.get_path('10740.shp'))

    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w3 = order(w, kmax = 3)
    >>> w3[1][0:5]
    [1, -1, 1, 2, 1]

    """

    ids = w.id_order
    info = {}
    for id_ in ids:
        s = [0] * w.n
        s[ids.index(id_)] = -1
        for j in w.neighbors[id_]:
            s[ids.index(j)] = 1
        k = 1
        while k < kmax:
            knext = k + 1
            if s.count(k):
                # get neighbors of order k
                js = [ids[j] for j, val in enumerate(s) if val == k]
                # get first order neighbors for order k neighbors
                for j in js:
                    next_neighbors = w.neighbors[j]
                    for neighbor in next_neighbors:
                        nid = ids.index(neighbor)
                        if s[nid] == 0:
                            s[nid] = knext
            k = knext
        info[id_] = s
    return info


def higher_order(w, k=2):
    """
    Contiguity weights object of order k.

    Parameters
    ----------

    w     : W
            spatial weights object
    k     : int
            order of contiguity

    Returns
    -------

    implicit : W
               spatial weights object

    Notes
    -----
    Proper higher order neighbors are returned such that i and j are k-order
    neighbors iff the shortest path from i-j is of length k.

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> w10 = lat2W(10, 10)
    >>> w10_2 = higher_order(w10, 2)
    >>> w10_2[0] ==  {2: 1.0, 11: 1.0, 20: 1.0}
    True
    >>> w5 = lat2W()
    >>> w5[0] ==  {1: 1.0, 5: 1.0}
    True
    >>> w5[1] == {0: 1.0, 2: 1.0, 6: 1.0}
    True
    >>> w5_2 = higher_order(w5,2)
    >>> w5_2[0] == {10: 1.0, 2: 1.0, 6: 1.0}
    True
    """
    return higher_order_sp(w, k)


def higher_order_sp(w, k=2, shortest_path=True, diagonal=False):
    """
    Contiguity weights for either a sparse W or W  for order k.

    Parameters
    ----------

    w           :   W
		    sparse_matrix, spatial weights object or scipy.sparse.csr.csr_instance

    k           :   int
                    Order of contiguity

    shortest_path :  boolean
                    True: i,j and k-order neighbors if the
                    shortest path for i,j is k
                    False: i,j are k-order neighbors if there
                    is a path from i,j of length k

    diagonal    :   boolean
                    True:  keep k-order (i,j) joins when i==j
                    False: remove k-order (i,j) joins when i==j

    Returns
    -------
    wk : W
	 WSP, type matches type of w argument

    Notes
    -----
    Lower order contiguities are removed.

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> import libpysal
    >>> w25 = lat2W(5,5)
    >>> w25.n
    25
    >>> w25[0] == {1: 1.0, 5: 1.0}
    True
    >>> w25_2 = libpysal.weights.util.higher_order_sp(w25, 2)
    >>> w25_2[0] == {10: 1.0, 2: 1.0, 6: 1.0}
    True
    >>> w25_2 = libpysal.weights.util.higher_order_sp(w25, 2, diagonal=True)
    >>> w25_2[0] ==  {0: 1.0, 10: 1.0, 2: 1.0, 6: 1.0}
    True
    >>> w25_3 = libpysal.weights.util.higher_order_sp(w25, 3)
    >>> w25_3[0] == {15: 1.0, 3: 1.0, 11: 1.0, 7: 1.0}
    True
    >>> w25_3 = libpysal.weights.util.higher_order_sp(w25, 3, shortest_path=False)
    >>> w25_3[0] == {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0, 11: 1.0, 15: 1.0}
    True

    """
    id_order = None
    if issubclass(type(w), W) or isinstance(w, W):
        if np.unique(np.hstack(list(w.weights.values()))) == np.array([1.0]):
            id_order = w.id_order
            w = w.sparse
        else:
            raise ValueError('Weights are not binary (0,1)')
    elif scipy.sparse.isspmatrix_csr(w):
        if not np.unique(w.data) == np.array([1.0]):
            raise ValueError('Sparse weights matrix is not binary (0,1) weights matrix.')
    else:
        raise TypeError("Weights provided are neither a binary W object nor "
                        "a scipy.sparse.csr_matrix")

    wk = w**k
    rk, ck = wk.nonzero()
    sk = set(zip(rk, ck))

    if shortest_path:
        for j in range(1, k):
            wj = w**j
            rj, cj = wj.nonzero()
            sj = set(zip(rj, cj))
            sk.difference_update(sj)

    if not diagonal:
        sk = set([(i,j) for i,j in sk if i!=j])

    if id_order:
        d = dict([(i,[]) for i in id_order])
        for pair in sk:
            k, v = pair
            k = id_order[k]
            v = id_order[v]
            d[k].append(v)
        return W(neighbors=d)
    else:
        d = {}
        for pair in sk:
            k, v = pair
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]
        return WSP(W(neighbors=d).sparse)


def w_local_cluster(w):
    """
    Local clustering coefficients for each unit as a node in a graph. [ws]_

    Parameters
    ----------

    w   : W
          spatial weights object

    Returns
    -------

    c     : array
            (w.n,1)
            local clustering coefficients

    Notes
    -----

    The local clustering coefficient :math:`c_i` quantifies how close the
    neighbors of observation :math:`i` are to being a clique:

            .. math::

               c_i = | \{w_{j,k}\} |/ (k_i(k_i - 1)): j,k \in N_i

    where :math:`N_i` is the set of neighbors to :math:`i`, :math:`k_i =
    |N_i|` and :math:`\{w_{j,k}\}` is the set of non-zero elements of the
    weights between pairs in :math:`N_i`. [Watts1998]_

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(3,3, rook=False)
    >>> w_local_cluster(w) 
    array([[1.        ],
           [0.6       ],
           [1.        ],
           [0.6       ],
           [0.42857143],
           [0.6       ],
           [1.        ],
           [0.6       ],
           [1.        ]])

    True

    """

    c = np.zeros((w.n, 1), float)
    w.transformation = 'b'
    for i, id in enumerate(w.id_order):
        ki = max(w.cardinalities[id], 1)  # deal with islands
        Ni = w.neighbors[id]
        wi = w_subset(w, Ni).full()[0]
        c[i] = wi.sum() / (ki * (ki - 1))
    return c


def shimbel(w):
    """
    Find the Shimbel matrix for first order contiguity matrix.

    Parameters
    ----------
    w     : W
            spatial weights object

    Returns
    -------

    info  : list
            list of lists; one list for each observation which stores
            the shortest order between it and each of the the other observations.

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> w5 = lat2W()
    >>> w5_shimbel = shimbel(w5)
    >>> w5_shimbel[0][24]
    8
    >>> w5_shimbel[0][0:4]
    [-1, 1, 2, 3]
    >>>
    """

    info = {}
    ids = w.id_order
    for i in ids:
        s = [0] * w.n
        s[ids.index(i)] = -1
        for j in w.neighbors[i]:
            s[ids.index(j)] = 1
        k = 1
        flag = s.count(0)
        while flag:
            p = -1
            knext = k + 1
            for j in range(s.count(k)):
                neighbor = s.index(k, p + 1)
                p = neighbor
                next_neighbors = w.neighbors[ids[p]]
                for neighbor in next_neighbors:
                    nid = ids.index(neighbor)
                    if s[nid] == 0:
                        s[nid] = knext
            k = knext
            flag = s.count(0)
        info[i] = s
    return info


def full(w):
    """
    Generate a full numpy array.

    Parameters
    ----------
    w        : W
               spatial weights object

    Returns
    -------
    (fullw, keys) : tuple
                    first element being the full numpy array and second element
                    keys being the ids associated with each row in the array.

    Examples
    --------
    >>> from libpysal.weights import lat2W, W
    >>> neighbors = {'first':['second'],'second':['first','third'],'third':['second']}
    >>> weights = {'first':[1],'second':[1,1],'third':[1]}
    >>> w = W(neighbors, weights)
    >>> wf, ids = full(w)
    >>> wf
    array([[0., 1., 0.],
           [1., 0., 1.],
           [0., 1., 0.]])

    >>> ids
    ['first', 'second', 'third']
    """
    return w.full()

def full2W(m, ids=None):
    '''
    Create a PySAL W object from a full array.

    Parameters
    ----------
    m       : array
              nxn array with the full weights matrix
    ids     : list
              User ids assumed to be aligned with m

    Returns
    -------
    w       : W
              PySAL weights object

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np

    Create an array of zeros

    >>> a = np.zeros((4, 4))

    For loop to fill it with random numbers

    >>> for i in range(len(a)):
    ...     for j in range(len(a[i])):
    ...         if i!=j:
    ...             a[i, j] = np.random.random(1)

    Create W object

    >>> w = libpysal.weights.util.full2W(a)
    >>> w.full()[0] == a
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]])

    Create list of user ids

    >>> ids = ['myID0', 'myID1', 'myID2', 'myID3']
    >>> w = libpysal.weights.util.full2W(a, ids=ids)
    >>> w.full()[0] == a
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]])
    '''
    if m.shape[0] != m.shape[1]:
        raise ValueError('Your array is not square')
    neighbors, weights = {}, {}
    for i in range(m.shape[0]):
    # for i, row in enumerate(m):
        row = m[i]
        if ids:
            i = ids[i]
        ngh = list(row.nonzero()[0])
        weights[i] = list(row[ngh])
        ngh = list(ngh)
        if ids:
            ngh = [ids[j] for j in ngh]
        neighbors[i] = ngh
    return W(neighbors, weights, id_order=ids)


def WSP2W(wsp, silence_warnings=False):

    """
    Convert a pysal WSP object (thin weights matrix) to a pysal W object.

    Parameters
    ----------
    wsp                     : WSP
                              PySAL sparse weights object
    silence_warnings   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------
    w       : W
              PySAL weights object

    Examples
    --------
    >>> from libpysal.weights import lat2W, WSP

    Build a 10x10 scipy.sparse matrix for a rectangular 2x5 region of cells
    (rook contiguity), then construct a PySAL sparse weights object (wsp).

    >>> sp = lat2SW(2, 5)
    >>> wsp = WSP(sp)
    >>> wsp.n
    10
    >>> wsp.sparse[0].todense()
    matrix([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=int8)

    Convert this sparse weights object to a standard PySAL weights object.

    >>> w = WSP2W(wsp)
    >>> w.n
    10
    >>> print(w.full()[0][0])
    [0. 1. 0. 0. 0. 1. 0. 0. 0. 0.]


    """
    wsp.sparse
    indices = wsp.sparse.indices
    data = wsp.sparse.data
    indptr = wsp.sparse.indptr
    id_order = wsp.id_order
    if id_order:
        # replace indices with user IDs
        indices = [id_order[i] for i in indices]
    else:
        id_order = list(range(wsp.n))
    neighbors, weights = {}, {}
    start = indptr[0]
    for i in range(wsp.n):
        oid = id_order[i]
        end = indptr[i + 1]
        neighbors[oid] = indices[start:end]
        weights[oid] = data[start:end]
        start = end
    ids = copy.copy(wsp.id_order)
    w = W(neighbors, weights, ids,
                silence_warnings=silence_warnings)
    w._sparse = copy.deepcopy(wsp.sparse)
    w._cache['sparse'] = w._sparse
    return w

def insert_diagonal(w, val=1.0, wsp=False):
    warn('This function is deprecated. Use fill_diagonal instead.')
    return fill_diagonal(w, val=val, wsp=wsp)

def fill_diagonal(w, val=1.0, wsp=False):
    """
    Returns a new weights object with values inserted along the main diagonal.

    Parameters
    ----------
    w        : W
               Spatial weights object

    diagonal : float, int or array
               Defines the value(s) to which the weights matrix diagonal should
               be set. If a constant is passed then each element along the
               diagonal will get this value (default is 1.0). An array of length
               w.n can be passed to set explicit values to each element along
               the diagonal (assumed to be in the same order as w.id_order).

    wsp      : boolean
               If True return a thin weights object of the type WSP, if False
               return the standard W object.

    Returns
    -------
    w        : W
               Spatial weights object

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> import numpy as np

    Build a basic rook weights matrix, which has zeros on the diagonal, then
    insert ones along the diagonal.

    >>> w = lat2W(5, 5, id_type='string')
    >>> w_const = insert_diagonal(w)
    >>> w['id0'] ==  {'id5': 1.0, 'id1': 1.0}
    True
    >>> w_const['id0'] == {'id5': 1.0, 'id0': 1.0, 'id1': 1.0}
    True

    Insert different values along the main diagonal.

    >>> diag = np.arange(100, 125)
    >>> w_var = insert_diagonal(w, diag)
    >>> w_var['id0'] == {'id5': 1.0, 'id0': 100.0, 'id1': 1.0}
    True

    """

    w_new = copy.deepcopy(w.sparse)
    w_new = w_new.tolil()
    if issubclass(type(val), np.ndarray):
        if w.n != val.shape[0]:
            raise Exception("shape of w and diagonal do not match")
        w_new.setdiag(val)
    elif isinstance(val, numbers.Number):
        w_new.setdiag([val] * w.n)
    else:
        raise Exception("Invalid value passed to diagonal")
    w_out = WSP(w_new, copy.copy(w.id_order))
    if wsp:
        return w_out
    else:
        return WSP2W(w_out)


def remap_ids(w, old2new, id_order=[]):
    """
    Remaps the IDs in a spatial weights object.

    Parameters
    ----------
    w        : W
               Spatial weights object

    old2new  : dictionary
               Dictionary where the keys are the IDs in w (i.e. "old IDs") and
               the values are the IDs to replace them (i.e. "new IDs")

    id_order : list
               An ordered list of new IDs, which defines the order of observations when
               iterating over W. If not set then the id_order in w will be
               used.

    Returns
    -------

    implicit : W
               Spatial weights object with new IDs

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> w = lat2W(3,2)
    >>> w.id_order
    [0, 1, 2, 3, 4, 5]
    >>> w.neighbors[0]
    [2, 1]
    >>> old_to_new = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}
    >>> w_new = remap_ids(w, old_to_new)
    >>> w_new.id_order
    ['a', 'b', 'c', 'd', 'e', 'f']
    >>> w_new.neighbors['a']
    ['c', 'b']

    """

    if not isinstance(w, W):
        raise Exception("w must be a spatial weights object")
    new_neigh = {}
    new_weights = {}
    for key, value in list(w.neighbors.items()):
        new_values = [old2new[i] for i in value]
        new_key = old2new[key]
        new_neigh[new_key] = new_values
        new_weights[new_key] = copy.copy(w.weights[key])
    if id_order:
        return W(new_neigh, new_weights, id_order)
    else:
        if w.id_order:
            id_order = [old2new[i] for i in w.id_order]
            return W(new_neigh, new_weights, id_order)
        else:
            return W(new_neigh, new_weights)


def get_ids(in_shps, idVariable):
    """
    Gets the IDs from the DBF file that moves with a given shape file or
    a geopandas.GeoDataFrame.

    Parameters
    ----------
    in_shps      : str or geopandas.GeoDataFrame
                   The input geographic data. Either
                   (1) a path to a shapefile including suffix (str); or
                   (2) a geopandas.GeoDataFrame.
    idVariable   : str
                   name of a column in the shapefile's DBF or the 
                   geopandas.GeoDataFrame to use for ids.

    Returns
    -------
    ids          : list
                   a list of IDs

    Examples
    --------
    >>> from libpysal.weights.util import get_ids
    >>> import libpysal
    >>> polyids = get_ids(libpysal.examples.get_path("columbus.shp"), "POLYID")
    >>> polyids[:5]
    [1, 2, 3, 4, 5]
    
    >>> from libpysal.weights.util import get_ids
    >>> import libpysal
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    >>> polyids = gdf["POLYID"]
    >>> polyids[:5]
    0    1
    1    2
    2    3
    3    4
    4    5
    Name: POLYID, dtype: int64
    
    """

    try:
        if type(in_shps) == str:
            dbname = os.path.splitext(in_shps)[0] + '.dbf'
            db = psopen(dbname)
            cols = db.header
            var = db.by_col[idVariable]
            db.close()
        else:
            cols = list(in_shps.columns)
            var = list(in_shps[idVariable])
        return var
    
    except IOError:
        msg = 'The shapefile "%s" appears to be missing its DBF file. '\
              + ' The DBF file "%s" could not be found.' % (in_shps, dbname)
        raise IOError(msg)
    except (AttributeError, KeyError):
        msg = 'The variable "%s" not found in the DBF/GDF. The the following '\
              + 'variables are present: %s.' % (idVariable, ','.join(cols))
        raise KeyError(msg)


def get_points_array(iterable):
    """
    Gets a data array of x and y coordinates from a given iterable
    Parameters
    ----------
    iterable      : iterable
                    arbitrary collection of shapes that supports iteration

    Returns
    -------
    points        : array
                    (n, 2)
                    a data array of x and y coordinates

    Notes
    -----
    If the given shape file includes polygons,
    this function returns x and y coordinates of the polygons' centroids

    """
    try:
        data = np.vstack([np.array(shape.centroid) for shape in iterable])
    except AttributeError:
        data = np.vstack([shape for shape in iterable])
    return data


def get_points_array_from_shapefile(shapefile):
    """
    Gets a data array of x and y coordinates from a given shapefile.

    Parameters
    ----------
    shapefile     : string
                    name of a shape file including suffix

    Returns
    -------
    points        : array
                    (n, 2)
                    a data array of x and y coordinates

    Notes
    -----
    If the given shape file includes polygons,
    this function returns x and y coordinates of the polygons' centroids

    Examples
    --------
    Point shapefile

    >>> import libpysal
    >>> from libpysal.weights.util import get_points_array_from_shapefile
    >>> xy = get_points_array_from_shapefile(libpysal.examples.get_path('juvenile.shp'))
    >>> xy[:3] 
    array([[94., 93.],
           [80., 95.],
           [79., 90.]])


    Polygon shapefile

    >>> xy = get_points_array_from_shapefile(libpysal.examples.get_path('columbus.shp'))
    >>> xy[:3]
    array([[ 8.82721847, 14.36907602],
           [ 8.33265837, 14.03162401],
           [ 9.01226541, 13.81971908]])
    """

    f = psopen(shapefile)
    data = get_points_array(f)
    return data


def min_threshold_distance(data, p=2):
    """
    Get the maximum nearest neighbor distance.

    Parameters
    ----------

    data    : array
              (n,k) or KDTree where KDtree.data is array (n,k)
              n observations on k attributes
    p       : float
              Minkowski p-norm distance metric parameter:
              1<=p<=infinity
              2: Euclidean distance
              1: Manhattan distance

    Returns
    -------
    nnd    : float
             maximum nearest neighbor distance between the n observations

    Examples
    --------
    >>> from libpysal.weights.util import min_threshold_distance
    >>> import numpy as np
    >>> x, y = np.indices((5, 5))
    >>> x.shape = (25, 1)
    >>> y.shape = (25, 1)
    >>> data = np.hstack([x, y])
    >>> min_threshold_distance(data)
    1.0

    """
    if issubclass(type(data), scipy.spatial.KDTree):
        kd = data
        data = kd.data
    else:
        kd = KDTree(data)
    nn = kd.query(data, k=2, p=p)
    nnd = nn[0].max(axis=0)[1]
    return nnd


def lat2SW(nrows=3, ncols=5, criterion="rook", row_st=False):
    """
    Create a sparse W matrix for a regular lattice.

    Parameters
    ----------

    nrows   : int
              number of rows
    ncols   : int
              number of columns
    rook    : {"rook", "queen", "bishop"}
              type of contiguity. Default is rook.
    row_st  : boolean
              If True, the created sparse W object is row-standardized so
              every row sums up to one. Defaults to False.

    Returns
    -------

    w : scipy.sparse.dia_matrix
        instance of a scipy sparse matrix

    Notes
    -----

    Observations are row ordered: first k observations are in row 0, next k in row 1, and so on.
    This method directly creates the W matrix using the strucuture of the contiguity type.

    Examples
    --------

    >>> from libpysal.weights import lat2W
    >>> w9 = lat2SW(3,3)
    >>> w9[0,1]
    1
    >>> w9[3,6]
    1
    >>> w9r = lat2SW(3,3, row_st=True)
    >>> w9r[3,6] == 1./3
    True
    """

    n = nrows * ncols
    diagonals = []
    offsets = []
    if criterion == "rook" or criterion == "queen":
        d = np.ones((1, n))
        for i in range(ncols - 1, n, ncols):
            d[0, i] = 0
        diagonals.append(d)
        offsets.append(-1)

        d = np.ones((1, n))
        diagonals.append(d)
        offsets.append(-ncols)

    if criterion == "queen" or criterion == "bishop":
        d = np.ones((1, n))
        for i in range(0, n, ncols):
            d[0, i] = 0
        diagonals.append(d)
        offsets.append(-(ncols - 1))

        d = np.ones((1, n))
        for i in range(ncols - 1, n, ncols):
            d[0, i] = 0
        diagonals.append(d)
        offsets.append(-(ncols + 1))
    data = np.concatenate(diagonals)
    offsets = np.array(offsets)
    m = sparse.dia_matrix((data, offsets), shape=(n, n), dtype=np.int8)
    m = m + m.T
    if row_st:
        m = sparse.spdiags(1. / m.sum(1).T, 0, *m.shape) * m
    return m


def write_gal(file, k=10):
    f = open(file, 'w')
    n = k * k
    f.write("0 %d" % n)
    for i in range(n):
        row = i / k
        col = i % k
        neighs = [i - i, i + 1, i - k, i + k]
        neighs = [j for j in neighs if j >= 0 and j < n]
        f.write("\n%d %d\n" % (i, len(neighs)))
        f.write(" ".join(map(str, neighs)))
    f.close()

def neighbor_equality(w1, w2):
    """
    Test if the neighbor sets are equal between two weights objects

    Parameters
    ----------

    w1 : W
        instance of spatial weights class W

    w2 : W
        instance of spatial weights class W

    Returns
    -------
    Boolean


    Notes
    -----
    Only set membership is evaluated, no check of the weight values is carried out.


    Examples
    --------
    >>> from libpysal.weights.util import neighbor_equality
    >>> from libpysal.weights import lat2W, W
    >>> w1 = lat2W(3,3)
    >>> w2 = lat2W(3,3)
    >>> neighbor_equality(w1, w2)
    True
    >>> w3 = lat2W(5,5)
    >>> neighbor_equality(w1, w3)
    False
    >>> n4 = w1.neighbors.copy()
    >>> n4[0] = [1]
    >>> n4[1] = [4, 2]
    >>> w4 = W(n4)
    >>> neighbor_equality(w1, w4)
    False
    >>> n5 = w1.neighbors.copy()
    >>> n5[0]
    [3, 1]
    >>> n5[0] = [1, 3]
    >>> w5 = W(n5)
    >>> neighbor_equality(w1, w5)
    True

    """
    n1 = w1.neighbors
    n2 = w2.neighbors
    ids_1 = set(n1.keys())
    ids_2 = set(n2.keys())
    if ids_1 != ids_2:
        return False
    for i in ids_1:
        if set(w1.neighbors[i]) != set(w2.neighbors[i]):
            return False
    return True

def isKDTree(obj):
    """
    This is a utility function to determine whether or not an object is a
    KDTree, since KDTree and cKDTree have no common parent type
    """
    return any([issubclass(type(obj), KDTYPE) for KDTYPE in KDTREE_TYPES])

def attach_islands(w, w_knn1):
    """
    Attach nearest neighbor to islands in spatial weight w.

    Parameters
    ----------

    w            : libpysal.weights.W
                   pysal spatial weight object (unstandardized).
    w_knn1       : libpysal.weights.W
                   Nearest neighbor pysal spatial weight object (k=1).

    Returns
    -------
                 : libpysal.weights.W
                   pysal spatial weight object w without islands.

    Examples
    --------
    >>> from libpysal.weights import lat2W
    >>> import libpysal
    >>> w = libpysal.weights.contiguity.Rook.from_shapefile(libpysal.examples.get_path('10740.shp'))
    >>> w.islands
    [163]
    >>> w_knn1 = libpysal.weights.distance.KNN.from_shapefile(libpysal.examples.get_path('10740.shp'),k=1)
    >>> w_attach = attach_islands(w, w_knn1)
    >>> w_attach.islands
    []
    >>> w_attach[w.islands[0]]
    {166: 1.0}

    """

    neighbors, weights = copy.deepcopy(w.neighbors), copy.deepcopy(w.weights)
    if not len(w.islands):
        print("There are no disconnected observations (no islands)!")
        return w
    else:
        for island in w.islands:
            nb = w_knn1.neighbors[island][0]
            if type(island) is float:
                nb = float(nb)
            neighbors[island] = [nb]
            weights[island] = [1.0]
            neighbors[nb] = neighbors[nb] + [island]
            weights[nb] = weights[nb] + [1.0]
        return W(neighbors, weights, id_order=w.id_order)

def nonplanar_neighbors(w, geodataframe, tolerance=0.001):
    """
    Detect neighbors for non-planar polygon collections


    Parameters
    ----------

    w:   pysal W
         A spatial weights object with reported islands


    geodataframe: GeoDataframe
                  The polygon dataframe from which w was constructed.

    tolerance: float
               The percentage of the minimum horizontal or vertical extent (minextent) of
               the dataframe to use in defining  a buffering distance to allow for fuzzy
               contiguity detection. The buffering distance is equal to tolerance*minextent.

    Attributes
    ----------

    non_planar_joins : dictionary
               Stores the new joins detected. Key is the id of the focal unit, value is a list of neighbor ids.

    Returns
    -------

    w: pysal W
       Spatial weights object that encodes fuzzy neighbors.
       This will have an attribute `non_planar_joins` to indicate what new joins were detected.

    Notes
    -----

    This relaxes the notion of contiguity neighbors for the case of shapefiles
    that violate the condition of planar enforcement. It handles three types
    of conditions present in such files that would result in islands when using
    the regular PySAL contiguity methods. The first are edges for nearby
    polygons that should be shared, but are digitized separately for the
    individual polygons and the resulting edges do not coincide, but instead
    the edges intersect. The second case is similar to the first, only the
    resultant edges do not intersect but are "close". The final case arises
    when one polygon is "inside" a second polygon but is not encoded to
    represent a hole in the containing polygon.

    The buffering check assumes the geometry coordinates are projected.

    Examples
    --------

    >>> import geopandas as gpd
    >>> import libpysal
    >>> df = gpd.read_file(libpysal.examples.get_path('map_RS_BR.shp'))
    >>> w = libpysal.weights.contiguity.Queen.from_dataframe(df)
    >>> import libpysal
    >>> w.islands
    [0, 4, 23, 27, 80, 94, 101, 107, 109, 119, 122, 139, 169, 175, 223, 239, 247, 253, 254, 255, 256, 261, 276, 291, 294, 303, 321, 357, 374]
    >>> wnp = libpysal.weights.util.nonplanar_neighbors(w, df)
    >>> wnp.islands
    []
    >>> w.neighbors[0]
    []
    >>> wnp.neighbors[0]
    [23, 59, 152, 239]
    >>> wnp.neighbors[23]
    [0, 45, 59, 107, 152, 185, 246]
    >>>

    Also see `nonplanarweights.ipynb`

    References
    ----------

    Planar Enforcement: http://ibis.geog.ubc.ca/courses/klink/gis.notes/ncgia/u12.html#SEC12.6


    """

    gdf = geodataframe
    assert gdf.sindex, 'GeoDataFrame must have a spatial index. Please make sure you have `libspatialindex` installed'
    islands = w.islands
    joins = copy.deepcopy(w.neighbors)
    candidates = gdf.geometry
    fixes = defaultdict(list)

    # first check for intersecting polygons
    for island in islands:
        focal = gdf.iloc[island].geometry
        neighbors = [j for j, candidate in enumerate(candidates) if focal.intersects(candidate) and j!= island]
        if len(neighbors) > 0:
            for neighbor in neighbors:
                if neighbor not in joins[island]:
                    fixes[island].append(neighbor)
                    joins[island].append(neighbor)
                if island not in joins[neighbor]:
                    fixes[neighbor].append(island)
                    joins[neighbor].append(island)

    # if any islands remain, dilate them and check for intersection
    if islands:
        x0,y0,x1,y1 = gdf.total_bounds
        distance = tolerance * min(x1-x0, y1-y0)
        for island in islands:
            dilated = gdf.iloc[island].geometry.buffer(distance)
            neighbors = [j for j, candidate in enumerate(candidates) if dilated.intersects(candidate) and j!= island]
            if len(neighbors) > 0:
                for neighbor in neighbors:
                    if neighbor not in joins[island]:
                        fixes[island].append(neighbor)
                        joins[island].append(neighbor)
                    if island not in joins[neighbor]:
                        fixes[neighbor].append(island)
                        joins[neighbor].append(island)

    w = W(joins)
    w.non_planar_joins = fixes
    return w

@requires('geopandas')
def fuzzy_contiguity(gdf, tolerance=0.005, buffering=False, drop=True):
    """
    Fuzzy contiguity spatial weights

    Parameters
    ----------

    gdf:   GeoDataFrame

    tolerance: float
               The percentage of the length of the minimum side of the bounding rectangle for the GeoDataFrame to use in determining the buffering distance.

    buffering: boolean
               If False (default) joins will only be detected for features that intersect (touch, contain, within).
               If True then features will be buffered and intersections will be based on buffered features.

    drop: boolean
          If True (default), the buffered features are removed from the GeoDataFrame. If False, buffered features are added to the GeoDataFrame.

    Returns
    -------

    w:  PySAL W
        Spatial weights based on fuzzy contiguity. Weights are binary.

    Examples
    --------

    >>> import libpysal as lps
    >>> import geopandas as gpd
    >>> rs = lps.examples.get_path('map_RS_BR.shp')
    >>> rs_df = gpd.read_file(rs)
    >>> wq = lps.weights.contiguity.Queen.from_dataframe(rs_df)
    >>> len(wq.islands)
    29
    >>> wq[0]
    {}
    >>> wf = fuzzy_contiguity(rs_df)
    >>> wf.islands
    []
    >>> wf[0] == dict({239: 1.0, 59: 1.0, 152: 1.0, 23: 1.0, 107: 1.0})
    True

    Example needing to use buffering

    >>> import libpysal as lps
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> p0 = Polygon([(0,0), (10,0), (10,10)])
    >>> p1 = Polygon([(10,1), (10,2), (15,2)])
    >>> p2 = Polygon([(12,2.001), (14, 2.001), (13,10)])
    >>> gs = gpd.GeoSeries([p0,p1,p2])
    >>> gdf = gpd.GeoDataFrame(geometry=gs)
    >>> wf = fuzzy_contiguity(gdf)
    >>> wf.islands
    [2]
    >>> wfb = fuzzy_contiguity(gdf, buffering=True)
    >>> wfb.islands
    []
    >>> wfb[2]
    {1: 1.0}

    Notes
    -----

    This relaxes the notion of contiguity neighbors for the case of feature
    collections that violate the condition of planar enforcement. It handles
    three types of conditions present in such collections that would result in
    islands when using the regular PySAL contiguity methods. The first are
    edges for nearby polygons that should be shared, but are digitized
    separately for the individual polygons and the resulting edges do not
    coincide, but instead the edges intersect. The second case is similar to
    the first, only the resultant edges do not intersect but are "close". The
    final case arises when one polygon is "inside" a second polygon but is not
    encoded to represent a hole in the containing polygon.

    Detection of the second case will require setting buffering=True and exploring different values for tolerance.

    The buffering check assumes the geometry coordinates are projected.


    References
    ----------

    Planar Enforcement: http://ibis.geog.ubc.ca/courses/klink/gis.notes/ncgia/u12.html#SEC12.6


    """
    if buffering:
        # buffer each shape
        minx, miny, maxx, maxy = gdf.total_bounds
        buffer = tolerance * 0.5 * abs(min(maxx-minx, maxy-miny))
        # create new geometry column
        new_geometry = gpd.GeoSeries([feature.buffer(buffer) for feature in gdf.geometry])
        gdf['_buffer'] = new_geometry
        old_geometry_name = gdf.geometry.name
        gdf.set_geometry('_buffer', inplace=True)
    assert gdf.sindex, 'GeoDataFrame must have a spatial index. Please make sure you have `libspatialindex` installed'
    tree = gdf.sindex
    neighbors = {}
    n,k = gdf.shape
    for i in range(n):
        geom = gdf.geometry.iloc[i]
        hits = list(tree.intersection(geom.bounds))
        possible = gdf.iloc[hits]
        ids = possible.intersects(geom).index.tolist()
        ids.remove(i)
        neighbors[i] = ids

    if buffering:
        gdf.set_geometry(old_geometry_name, inplace=True)
        if drop:
            gdf.drop(columns=['_buffer'], inplace=True)

    return W(neighbors)



if __name__ == "__main__":

    from pysal import lat2W
    assert (lat2W(5, 5).sparse.todense() == lat2SW(5, 5).todense()).all()
    assert (lat2W(5, 3).sparse.todense() == lat2SW(5, 3).todense()).all()
    assert (lat2W(5, 3, rook=False).sparse.todense() == lat2SW(5, 3,
                                                               'queen').todense()).all()
    assert (lat2W(50, 50, rook=False).sparse.todense() == lat2SW(50,

                                                                 50, 'queen').todense()).all()
