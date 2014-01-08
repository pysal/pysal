import pysal
from pysal.common import *
import pysal.weights
import numpy as np
from scipy import sparse, float32
import scipy.spatial
import os
import operator

__all__ = ['lat2W', 'regime_weights', 'comb', 'order', 'higher_order', 'shimbel', 'remap_ids', 'full2W', 'full', 'WSP2W', 'insert_diagonal', 'get_ids', 'get_points_array_from_shapefile', 'min_threshold_distance', 'lat2SW', 'w_local_cluster', 'higher_order_sp']


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
    '0.296'
    >>> w9[0]
    {1: 1.0, 3: 1.0}
    >>> w9[3]
    {0: 1.0, 4: 1.0, 6: 1.0}
    >>>
    """
    n = nrows * ncols
    r1 = nrows - 1
    c1 = ncols - 1
    rid = [i / ncols for i in xrange(n)]
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
    return pysal.weights.W(w, weights, ids)


def regime_weights(regimes):
    """
    Construct spatial weights for regime neighbors.

    Block contiguity structures are relevant when defining neighbor relations
    based on membership in a regime. For example, all counties belonging to
    the same state could be defined as neighbors, in an analysis of all
    counties in the US.

    Parameters
    ----------
    regimes : list or array
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
    >>> w.weights[0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> w.neighbors[0]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
    >>> regimes = ['n','n','s','s','e','e','w','w','e']
    >>> n = len(regimes)
    >>> w = regime_weights(regimes)
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
    return pysal.weights.W(neighbors)


def comb(items, n=None):
    """
    Combinations of size n taken from items

    Parameters
    ----------

    items : sequence
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
    Implements the algorithm in Anselin and Smirnov (1996) [1]_


    Examples
    --------
    >>> from pysal import rook_from_shapefile as rfs
    >>> w = rfs(pysal.examples.get_path('10740.shp'))
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w3 = order(w, kmax = 3)
    >>> w3[1][0:5]
    [1, -1, 1, 2, 1]

    References
    ----------
    .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
       constructing proper higher order spatial lag operators. Journal of
       Regional Science, 36, 67-89.

    """
    ids = w.neighbors.keys()
    info = {}
    for id in ids:
        s = [0] * w.n
        s[ids.index(id)] = -1
        for j in w.neighbors[id]:
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
        info[id] = s
    return info


def higher_order(w, k=2):
    """
    Contiguity weights object of order k

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
    Implements the algorithm in Anselin and Smirnov (1996) [1]_

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
    {2: 1.0, 10: 1.0, 6: 1.0}

    References
    ----------
    .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
       constructing proper higher order spatial lag operators. Journal of
       Regional Science, 36, 67-89.
    """
    info = order(w, k)
    ids = info.keys()
    neighbors = {}
    weights = {}
    for id in ids:
        nids = [ids[j] for j, o in enumerate(info[id]) if o == k]
        neighbors[id] = nids
        weights[id] = [1.0] * len(nids)
    return pysal.weights.W(neighbors, weights)

def higher_order_sp(wsp, k=2):
    """
    Contiguity weights for a sparse W for order k

    Arguments
    =========

    wsp:  WSP instance

    k: Order of contiguity

    Return
    ------

    wk: WSP instance
        binary sparse contiguity of order k

    Notes
    -----
    Lower order contiguities are removed.

    Examples
    -------

    >>> import pysal
    >>> w25 = pysal.lat2W(5,5)
    >>> w25.n
    25
    >>> ws25 = w25.sparse
    >>> ws25o3 = pysal.weights.higher_order_sp(ws25,3)
    >>> w25o3 = pysal.weights.higher_order(w25,3)
    >>> w25o3[12]
    {1: 1.0, 3: 1.0, 5: 1.0, 9: 1.0, 15: 1.0, 19: 1.0, 21: 1.0, 23: 1.0}
    >>> pysal.weights.WSP2W(ws25o3)[12]
    {1: 1.0, 3: 1.0, 5: 1.0, 9: 1.0, 15: 1.0, 19: 1.0, 21: 1.0, 23: 1.0}
    >>>     
    """


    wk = wsp**k
    rk,ck = wk.nonzero()
    sk = set(zip(rk,ck))
    for j in range(1,k):
        wj = wsp**j
        rj,cj = wj.nonzero()
        sj = set(zip(rj,cj))
        sk.difference_update(sj)
    d= {}
    for pair in sk:
        k,v = pair
        if d.has_key(k):
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

    c     : array (w.n,1)
            local clustering coefficients


    Notes
    -----

    The local clustering coefficient :math:`c_i` quantifies how close the
    neighbors of observation :math:`i` are to being a clique:

            .. math::

               c_i = | \{w_{j,k}\} |/ (k_i(k_i - 1)): j,k \in N_i

    where :math:`N_i` is the set of neighbors to :math:`i`, :math:`k_i =
    |N_i|` and :math:`\{w_{j,k}\}` is the set of non-zero elements of the
    weights between pairs in :math:`N_i`.


    References
    ----------

    .. [ws] Watts, D.J. and S.H. Strogatz (1988) "Collective dynamics of 'small-world' networks". Nature, 393: 440-442.



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

    info  : list of lists
            one list for each observation which stores the shortest
            order between it and each of the the other observations.

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
    for id in ids:
        s = [0] * w.n
        s[ids.index(id)] = -1
        for j in w.neighbors[id]:
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
        info[id] = s
    return info


def full(w):
    """
    Generate a full numpy array

    Parameters
    ----------
    w        : W
               spatial weights object

    Returns
    -------

    implicit : tuple
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
    Create a PySAL W object from a full array
    ...

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
    #for i, row in enumerate(m):
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
    w = pysal.W(neighbors, weights, ids, silent_island_warning=silent_island_warning)
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
    Remaps the IDs in a spatial weights object

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
    Gets the IDs from the DBF file that moves with a given shape file

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
        msg = 'The shapefile "%s" appears to be missing its DBF file. The DBF file "%s" could not be found.' % (shapefile, dbname)
        raise IOError(msg)
    except AttributeError:
        msg = 'The variable "%s" was not found in the DBF file. The DBF contains the following variables: %s.' % (idVariable, ','.join(db.header))
        raise KeyError(msg)


def get_points_array_from_shapefile(shapefile):
    """
    Gets a data array of x and y coordinates from a given shape file

    Parameters
    ----------
    shapefile     : string
                    name of a shape file including suffix

    Returns
    -------
    points        : array (n,2)
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


def min_threshold_distance(data,p=2):
    """
    Get the maximum nearest neighbor distance

    Parameters
    ----------

    data    : array (n,k) or KDTree where KDtree.data is array (n,k)
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
    rook    : "rook", "queen", or "bishop"
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

if __name__ == "__main__":
    from pysal import lat2W

    assert (lat2W(5, 5).sparse.todense() == lat2SW(5, 5).todense()).all()
    assert (lat2W(5, 3).sparse.todense() == lat2SW(5, 3).todense()).all()
    assert (lat2W(5, 3, rook=False).sparse.todense() == lat2SW(5, 3,
                                                               'queen').todense()).all()
    assert (lat2W(50, 50, rook=False).sparse.todense() == lat2SW(50,
                                                                 50, 'queen').todense()).all()
