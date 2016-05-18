import pysal
from pysal.common import *
import pysal.weights
import numpy as np
from scipy import sparse, float32
import scipy.spatial
import os
import operator
import scipy

__all__ = ['lat2W', 'block_weights', 'comb', 'order', 'higher_order',
           'shimbel', 'remap_ids', 'full2W', 'full', 'WSP2W',
           'insert_diagonal', 'get_ids', 'get_points_array_from_shapefile',
           'min_threshold_distance', 'lat2SW', 'w_local_cluster',
           'higher_order_sp', 'hexLat2W', 'regime_weights']


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

    >>> import pysal as ps
    >>> w = ps.lat2W()
    >>> w.neighbors[1]
    [0, 6, 2]
    >>> w.neighbors[21]
    [16, 20, 22]
    >>> wh = ps.hexLat2W()
    >>> wh.neighbors[1]
    [0, 6, 2, 5, 7]
    >>> wh.neighbors[21]
    [16, 20, 22]
    >>>
    """

    if nrows == 1 or ncols == 1:
        print "Hexagon lattice requires at least 2 rows and columns"
        print "Returning a linear contiguity structure"
        return lat2W(nrows, ncols)

    n = nrows * ncols
    rid = [i // ncols for i in xrange(n)]
    cid = [i % ncols for i in xrange(n)]
    r1 = nrows - 1
    c1 = ncols - 1

    w = lat2W(nrows, ncols).neighbors
    for i in xrange(n):
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


    return pysal.weights.W(w)


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

    >>> from pysal import lat2W
    >>> w9 = lat2W(3,3)
    >>> "%.3f"%w9.pct_nonzero
    '29.630'
    >>> w9[0]
    {1: 1.0, 3: 1.0}
    >>> w9[3]
    {0: 1.0, 4: 1.0, 6: 1.0}
    >>>
    """
    n = nrows * ncols
    r1 = nrows - 1
    c1 = ncols - 1
    rid = [i // ncols for i in xrange(n)] #must be floor!
    cid = [i % ncols for i in xrange(n)]
    w = {}
    r = below = 0
    for i in xrange(n - 1):
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
    ids = range(n)
    if id_type == 'string':
        ids = ['id' + str(i) for i in ids]
    elif id_type == 'float':
        ids = [i * 1. for i in ids]
    if id_type == 'string' or id_type == 'float':
        id_dict = dict(zip(range(n), ids))
        alt_w = {}
        alt_weights = {}
        for i in w:
            values = [id_dict[j] for j in w[i]]
            key = id_dict[i]
            alt_w[key] = values
            alt_weights[key] = weights[i]
        w = alt_w
        weights = alt_weights
    return pysal.weights.W(w, weights, ids=ids, id_order=ids[:])

def regime_weights(regimes):
    """
    Construct spatial weights for regime neighbors.

    Block contiguity structures are relevant when defining neighbor relations
    based on membership in a regime. For example, all counties belonging to
    the same state could be defined as neighbors, in an analysis of all
    counties in the US.

    Parameters
    ----------
    regimes : array, list
              ids of which regime an observation belongs to

    Returns
    -------

    W : spatial weights instance

    Examples
    --------

    >>> from pysal import regime_weights
    >>> import numpy as np
    >>> regimes = np.ones(25)
    >>> regimes[range(10,20)] = 2
    >>> regimes[range(21,25)] = 3
    >>> regimes
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,
            2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  3.,  3.,  3.,  3.])
    >>> w = regime_weights(regimes)
    PendingDepricationWarning: regime_weights will be renamed to block_weights in PySAL 2.0
    >>> w.weights[0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> w.neighbors[0]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
    >>> regimes = ['n','n','s','s','e','e','w','w','e']
    >>> n = len(regimes)
    >>> w = regime_weights(regimes)
    PendingDepricationWarning: regime_weights will be renamed to block_weights in PySAL 2.0
    >>> w.neighbors
    {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}

    Notes
    -----
    regime_weights will be deprecated in PySAL 2.0 and renamed to block_weights.

    """
    msg = "PendingDepricationWarning: regime_weights will be "
    msg += "renamed to block_weights in PySAL 2.0"
    print msg
    return block_weights(regimes)



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

    >>> from pysal import block_weights
    >>> import numpy as np
    >>> regimes = np.ones(25)
    >>> regimes[range(10,20)] = 2
    >>> regimes[range(21,25)] = 3
    >>> regimes
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,
            2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  3.,  3.,  3.,  3.])
    >>> w = block_weights(regimes)
    >>> w.weights[0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> w.neighbors[0]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
    >>> regimes = ['n','n','s','s','e','e','w','w','e']
    >>> n = len(regimes)
    >>> w = block_weights(regimes)
    >>> w.neighbors
    {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}
    """
    rids = np.unique(regimes)
    neighbors = {}
    NPNZ = np.nonzero
    regimes = np.array(regimes)
    for rid in rids:
        members = NPNZ(regimes == rid)[0]
        for member in members:
            neighbors[member] = members[NPNZ(members != member)[0]].tolist()
    w = pysal.weights.W(neighbors)
    if ids is not None:
        w.remap_ids(ids)
    if sparse:
        w = pysal.weights.WSP(w.sparse, id_order=ids)
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
    ...     print c
    ...
    [0, 1]
    [0, 2]
    [0, 3]
    [1, 2]
    [1, 3]
    [2, 3]

    """
    if n is None:
        n = len(items)
    for i in range(len(items)):
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
    >>> from pysal import rook_from_shapefile as rfs
    >>> w = rfs(pysal.examples.get_path('10740.shp'))
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w3 = order(w, kmax = 3)
    >>> w3[1][0:5]
    [1, -1, 1, 2, 1]

    """
    #ids = w.neighbors.keys()
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
    >>> from pysal import lat2W, higher_order
    >>> w10 = lat2W(10, 10)
    >>> w10_2 = higher_order(w10, 2)
    >>> w10_2[0]
    {2: 1.0, 11: 1.0, 20: 1.0}
    >>> w5 = lat2W()
    >>> w5[0]
    {1: 1.0, 5: 1.0}
    >>> w5[1]
    {0: 1.0, 2: 1.0, 6: 1.0}
    >>> w5_2 = higher_order(w5,2)
    >>> w5_2[0]
    {10: 1.0, 2: 1.0, 6: 1.0}
    """
    return higher_order_sp(w, k)


def higher_order_sp(w, k=2, shortest_path=True, diagonal=False):
    """
    Contiguity weights for either a sparse W or pysal.weights.W  for order k.

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

    >>> import pysal
    >>> w25 = pysal.lat2W(5,5)
    >>> w25.n
    25
    >>> w25[0]
    {1: 1.0, 5: 1.0}
    >>> w25_2 = pysal.weights.util.higher_order_sp(w25, 2)
    >>> w25_2[0]
    {10: 1.0, 2: 1.0, 6: 1.0}
    >>> w25_2 = pysal.weights.util.higher_order_sp(w25, 2, diagonal=True)
    >>> w25_2[0]
    {0: 1.0, 10: 1.0, 2: 1.0, 6: 1.0}
    >>> w25_3 = pysal.weights.util.higher_order_sp(w25, 3)
    >>> w25_3[0]
    {15: 1.0, 3: 1.0, 11: 1.0, 7: 1.0}
    >>> w25_3 = pysal.weights.util.higher_order_sp(w25, 3, shortest_path=False)
    >>> w25_3[0]
    {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0, 11: 1.0, 15: 1.0}

    """

    tw = type(w)
    id_order = None
    if tw == pysal.weights.weights.W:
        id_order = w.id_order
        w = w.sparse
    elif tw != scipy.sparse.csr.csr_matrix:
        print "Unsupported sparse argument."
        return None

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
        return pysal.W(neighbors=d)
    else:
        d = {}
        for pair in sk:
            k, v = pair
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]
        return pysal.weights.WSP(pysal.W(neighbors=d).sparse)


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

    >>> w = pysal.lat2W(3,3, rook=False)
    >>> w_local_cluster(w)
    array([[ 1.        ],
           [ 0.6       ],
           [ 1.        ],
           [ 0.6       ],
           [ 0.42857143],
           [ 0.6       ],
           [ 1.        ],
           [ 0.6       ],
           [ 1.        ]])

    """

    c = np.zeros((w.n, 1), float)
    w.transformation = 'b'
    for i, id in enumerate(w.id_order):
        ki = max(w.cardinalities[id], 1)  # deal with islands
        Ni = w.neighbors[id]
        wi = pysal.w_subset(w, Ni).full()[0]
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
    >>> from pysal import lat2W, shimbel
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
    >>> from pysal import W, full
    >>> neighbors = {'first':['second'],'second':['first','third'],'third':['second']}
    >>> weights = {'first':[1],'second':[1,1],'third':[1]}
    >>> w = W(neighbors, weights)
    >>> wf, ids = full(w)
    >>> wf
    array([[ 0.,  1.,  0.],
           [ 1.,  0.,  1.],
           [ 0.,  1.,  0.]])
    >>> ids
    ['first', 'second', 'third']
    """
    wfull = np.zeros([w.n, w.n], dtype=float)
    keys = w.neighbors.keys()
    if w.id_order:
        keys = w.id_order
    for i, key in enumerate(keys):
        n_i = w.neighbors[key]
        w_i = w.weights[key]
        for j, wij in zip(n_i, w_i):
            c = keys.index(j)
            wfull[i, c] = wij
    return (wfull, keys)


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
    >>> import pysal as ps
    >>> import numpy as np

    Create an array of zeros

    >>> a = np.zeros((4, 4))

    For loop to fill it with random numbers

    >>> for i in range(len(a)):
    ...     for j in range(len(a[i])):
    ...         if i!=j:
    ...             a[i, j] = np.random.random(1)

    Create W object

    >>> w = ps.weights.util.full2W(a)
    >>> w.full()[0] == a
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]], dtype=bool)

    Create list of user ids

    >>> ids = ['myID0', 'myID1', 'myID2', 'myID3']
    >>> w = ps.weights.util.full2W(a, ids=ids)
    >>> w.full()[0] == a
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]], dtype=bool)
    '''
    if m.shape[0] != m.shape[1]:
        raise ValueError('Your array is not square')
    neighbors, weights = {}, {}
    for i in xrange(m.shape[0]):
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
    return pysal.W(neighbors, weights, id_order=ids)


def WSP2W(wsp, silent_island_warning=False):

    """
    Convert a pysal WSP object (thin weights matrix) to a pysal W object.

    Parameters
    ----------
    wsp                     : WSP
                              PySAL sparse weights object
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements
                              for every observation with islands

    Returns
    -------
    w       : W
              PySAL weights object

    Examples
    --------
    >>> import pysal

    Build a 10x10 scipy.sparse matrix for a rectangular 2x5 region of cells
    (rook contiguity), then construct a PySAL sparse weights object (wsp).

    >>> sp = pysal.weights.lat2SW(2, 5)
    >>> wsp = pysal.weights.WSP(sp)
    >>> wsp.n
    10
    >>> print wsp.sparse[0].todense()
    [[0 1 0 0 0 1 0 0 0 0]]

    Convert this sparse weights object to a standard PySAL weights object.

    >>> w = pysal.weights.WSP2W(wsp)
    >>> w.n
    10
    >>> print w.full()[0][0]
    [ 0.  1.  0.  0.  0.  1.  0.  0.  0.  0.]

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
        id_order = range(wsp.n)
    neighbors, weights = {}, {}
    start = indptr[0]
    for i in xrange(wsp.n):
        oid = id_order[i]
        end = indptr[i + 1]
        neighbors[oid] = indices[start:end]
        weights[oid] = data[start:end]
        start = end
    ids = copy.copy(wsp.id_order)
    w = pysal.W(neighbors, weights, ids,
                silent_island_warning=silent_island_warning)
    w._sparse = copy.deepcopy(wsp.sparse)
    w._cache['sparse'] = w._sparse
    return w


def insert_diagonal(w, diagonal=1.0, wsp=False):
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
    >>> import pysal
    >>> import numpy as np

    Build a basic rook weights matrix, which has zeros on the diagonal, then
    insert ones along the diagonal.

    >>> w = pysal.lat2W(5, 5, id_type='string')
    >>> w_const = pysal.weights.insert_diagonal(w)
    >>> w['id0']
    {'id5': 1.0, 'id1': 1.0}
    >>> w_const['id0']
    {'id5': 1.0, 'id0': 1.0, 'id1': 1.0}

    Insert different values along the main diagonal.

    >>> diag = np.arange(100, 125)
    >>> w_var = pysal.weights.insert_diagonal(w, diag)
    >>> w_var['id0']
    {'id5': 1.0, 'id0': 100.0, 'id1': 1.0}

    """

    w_new = copy.deepcopy(w.sparse)
    w_new = w_new.tolil()
    if issubclass(type(diagonal), np.ndarray):
        if w.n != diagonal.shape[0]:
            raise Exception("shape of w and diagonal do not match")
        w_new.setdiag(diagonal)
    elif operator.isNumberType(diagonal):
        w_new.setdiag([diagonal] * w.n)
    else:
        raise Exception("Invalid value passed to diagonal")
    w_out = pysal.weights.WSP(w_new, copy.copy(w.id_order))
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
    >>> from pysal import lat2W, remap_ids
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

    if not isinstance(w, pysal.weights.W):
        raise Exception("w must be a spatial weights object")
    new_neigh = {}
    new_weights = {}
    for key, value in w.neighbors.iteritems():
        new_values = [old2new[i] for i in value]
        new_key = old2new[key]
        new_neigh[new_key] = new_values
        new_weights[new_key] = copy.copy(w.weights[key])
    if id_order:
        return pysal.weights.W(new_neigh, new_weights, id_order)
    else:
        if w.id_order:
            id_order = [old2new[i] for i in w.id_order]
            return pysal.weights.W(new_neigh, new_weights, id_order)
        else:
            return pysal.weights.W(new_neigh, new_weights)


def get_ids(shapefile, idVariable):
    """
    Gets the IDs from the DBF file that moves with a given shape file.

    Parameters
    ----------
    shapefile    : string
                   name of a shape file including suffix
    idVariable   : string
                   name of a column in the shapefile's DBF to use for ids

    Returns
    -------
    ids          : list
                   a list of IDs

    Examples
    --------
    >>> from pysal.weights.util import get_ids
    >>> polyids = get_ids(pysal.examples.get_path("columbus.shp"), "POLYID")
    >>> polyids[:5]
    [1, 2, 3, 4, 5]
    """

    try:
        dbname = os.path.splitext(shapefile)[0] + '.dbf'
        db = pysal.open(dbname)
        var = db.by_col[idVariable]
        db.close()
        return var
    except IOError:
        msg = 'The shapefile "%s" appears to be missing its DBF file. The DBF file "%s" could not be found.' % (
            shapefile, dbname)
        raise IOError(msg)
    except AttributeError:
        msg = 'The variable "%s" was not found in the DBF file. The DBF contains the following variables: %s.' % (
            idVariable, ','.join(db.header))
        raise KeyError(msg)


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

    >>> from pysal.weights.util import get_points_array_from_shapefile
    >>> xy = get_points_array_from_shapefile(pysal.examples.get_path('juvenile.shp'))
    >>> xy[:3]
    array([[ 94.,  93.],
           [ 80.,  95.],
           [ 79.,  90.]])

    Polygon shapefile

    >>> xy = get_points_array_from_shapefile(pysal.examples.get_path('columbus.shp'))
    >>> xy[:3]
    array([[  8.82721847,  14.36907602],
           [  8.33265837,  14.03162401],
           [  9.01226541,  13.81971908]])
    """

    f = pysal.open(shapefile)
    if f.type.__name__ == 'Polygon':
        data = np.array([shape.centroid for shape in f])
    elif f.type.__name__ == 'Point':
        data = np.array([shape for shape in f])
    f.close()
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
    >>> from pysal.weights.util import min_threshold_distance
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

    >>> from pysal import weights
    >>> w9 = weights.lat2SW(3,3)
    >>> w9[0,1]
    1
    >>> w9[3,6]
    1
    >>> w9r = weights.lat2SW(3,3, row_st=True)
    >>> w9r[3,6]
    0.33333333333333331
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
    for i in xrange(n):
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
    >>> from pysal.weights.util import neighbor_equality
    >>> w1 = pysal.lat2W(3,3)
    >>> w2 = pysal.lat2W(3,3)
    >>> neighbor_equality(w1, w2)
    True
    >>> w3 = pysal.lat2W(5,5)
    >>> neighbor_equality(w1, w3)
    False
    >>> n4 = w1.neighbors.copy()
    >>> n4[0] = [1]
    >>> n4[1] = [4, 2]
    >>> w4 = pysal.W(n4)
    >>> neighbor_equality(w1, w4)
    False
    >>> n5 = w1.neighbors.copy()
    >>> n5[0]
    [3, 1]
    >>> n5[0] = [1, 3]
    >>> w5 = pysal.W(n5)
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

if __name__ == "__main__":
    from pysal import lat2W

    assert (lat2W(5, 5).sparse.todense() == lat2SW(5, 5).todense()).all()
    assert (lat2W(5, 3).sparse.todense() == lat2SW(5, 3).todense()).all()
    assert (lat2W(5, 3, rook=False).sparse.todense() == lat2SW(5, 3,
                                                               'queen').todense()).all()
    assert (lat2W(50, 50, rook=False).sparse.todense() == lat2SW(50,

                                                                 50, 'queen').todense()).all()
