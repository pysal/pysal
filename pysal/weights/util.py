import pysal
from pysal.common import *
import pysal.weights
import numpy as np
from scipy import sparse,float32
from scipy.spatial import KDTree
import os, gc

__all__ = ['lat2W','regime_weights','comb','order', 'higher_order', 'shimbel', 'remap_ids','full','get_ids', 'min_threshold_distance','lat2SW']


def lat2W(nrows=5,ncols=5,rook=True,id_type='int'):
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
    >>> w9=lat2W(3,3)
    >>> w9.pct_nonzero
    0.29629629629629628
    >>> w9[0]
    {1: 1.0, 3: 1.0}
    >>> w9[3]
    {0: 1.0, 4: 1.0, 6: 1.0}
    >>> 
    """
    gc.disable()
    n=nrows*ncols
    r1=nrows-1
    c1=ncols-1
    rid=[ i/ncols for i in xrange(n) ]
    cid=[ i%ncols for i in xrange(n) ]
    w={}
    r=below=0
    for i in xrange(n-1):
        if rid[i]<r1:
            below=rid[i]+1
            r=below*ncols+cid[i]
            w[i]=w.get(i,[])+[r]
            w[r]=w.get(r,[])+[i]
        if cid[i]<c1:
            right=cid[i]+1
            c=rid[i]*ncols+right
            w[i]=w.get(i,[])+[c]
            w[c]=w.get(c,[])+[i]
        if not rook:
            # southeast bishop
            if cid[i]<c1 and rid[i]<r1:
                r=(rid[i]+1)*ncols + 1 + cid[i]
                w[i]=w.get(i,[])+[r]
                w[r]=w.get(r,[])+[i]
            # southwest bishop
            if cid[i]>0 and rid[i]<r1:
                r=(rid[i]+1)*ncols - 1 + cid[i]
                w[i]=w.get(i,[])+[r]
                w[r]=w.get(r,[])+[i]

    neighbors={}
    weights={}
    for key in w:
        weights[key]=[1.]*len(w[key])
    ids = range(n)
    if id_type=='string':
        ids = ['id'+str(i) for i in ids]
    elif id_type=='float':
        ids = [i*1. for i in ids]
    if id_type=='string' or id_type=='float':
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
    gc.enable()
    return pysal.weights.W(w,weights,ids)


def regime_weights(regimes):
    """
    >>> from pysal import regime_weights 
    >>> import numpy as np
    >>> regimes=np.ones(25)
    >>> regimes[range(10,20)]=2
    >>> regimes[range(21,25)]=3
    >>> regimes
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,
            2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  3.,  3.,  3.,  3.])
    >>> w=regime_weights(regimes)
    >>> w.weights[0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> w.neighbors[0]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
    >>> regimes=['n','n','s','s','e','e','w','w','e']
    >>> n=len(regimes)
    >>> w=regime_weights(regimes)
    >>> w.neighbors
    {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}
    """ 
    rids = np.unique(regimes)
    neighbors = {}
    NPNZ = np.nonzero
    regimes = np.array(regimes)
    for rid in rids:
        members = NPNZ(regimes==rid)[0]
        for member in members:
            neighbors[member] = members[NPNZ(members!=member)[0]].tolist()
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
    >>> x=range(4)
    >>> for c in comb(x,2):
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
        n=len(items)
    for i in range(len(items)):
        v=items[i:i+1]
        if n==1:
            yield v
        else:
            rest = items[i+1:]
            for c in comb(rest, n-1):
                yield v + c

def order(w,kmax=3):
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
    >>> w=rfs('../examples/10740.shp')
    >>> w3=order(w,kmax=3)
    >>> w3[1][0:5]
    [1, -1, 1, 2, 1]

    References
    ----------
    .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
       constructing proper higher order spatial lag operators. Journal of
       Regional Science, 36, 67-89. 

    """
    ids=w.neighbors.keys()
    info={}
    for id in ids:
        s=[0]*w.n
        s[ids.index(id)]=-1
        for j in w.neighbors[id]:
            s[ids.index(j)]=1
        k=1
        while k < kmax:
            knext=k+1
            if s.count(k):
                # get neighbors of order k
                js=[ids[j] for j,val in enumerate(s) if val==k]
                # get first order neighbors for order k neighbors
                for j in js:
                    next_neighbors=w.neighbors[j]
                    for neighbor in next_neighbors:
                        nid=ids.index(neighbor)
                        if s[nid]==0:
                            s[nid]=knext
            k=knext
        info[id]=s
    return info

def higher_order(w,order=2):
    """
    Contiguity weights object of order k 

    Parameters
    ----------

    w     : W
            spatial weights object
    order : int
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
    >>> w10=lat2W(10,10)
    >>> w10_2=higher_order(w10,2)
    >>> w10_2[0]
    {2: 1.0, 11: 1.0, 20: 1.0}
    >>> w5=lat2W()
    >>> w5[0]
    {1: 1.0, 5: 1.0}
    >>> w5[1]
    {0: 1.0, 2: 1.0, 6: 1.0}
    >>> w5_2=higher_order(w5,2)
    >>> w5_2[0]
    {2: 1.0, 10: 1.0, 6: 1.0}

    References
    ----------
    .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
       constructing proper higher order spatial lag operators. Journal of
       Regional Science, 36, 67-89. 
    """
    info=w.order(order)
    ids=info.keys()
    neighbors={}
    weights={}
    for id in ids:
        nids=[ids[j] for j,k in enumerate(info[id]) if order==k]
        neighbors[id]=nids
        weights[id]=[1.0]*len(nids)
    return pysal.weights.W(neighbors,weights)


def shimbel(w):
    """
    Find the Shmibel matrix for first order contiguity matrix.

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
    >>> w5=lat2W()
    >>> w5_shimbel=shimbel(w5)
    >>> w5_shimbel[0][24]
    8
    >>> w5_shimbel[0][0:4]
    [-1, 1, 2, 3]
    >>>
    """
    info={}
    ids=w.id_order
    for id in ids:
        s=[0]*w.n
        s[ids.index(id)]=-1
        for j in w.neighbors[id]:
            s[ids.index(j)]=1
        k=1
        flag=s.count(0)
        while flag:
            p=-1
            knext=k+1
            for j in range(s.count(k)):
                neighbor=s.index(k,p+1)
                p=neighbor
                next_neighbors=w.neighbors[ids[p]]
                for neighbor in next_neighbors:
                    nid=ids.index(neighbor)
                    if s[nid]==0:
                        s[nid]=knext
            k=knext
            flag=s.count(0)
        info[id]=s
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
    >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
    >>> weights={'first':[1],'second':[1,1],'third':[1]}
    >>> w=W(neighbors,weights)
    >>> wf,ids=full(w)
    >>> wf
    array([[ 0.,  1.,  0.],
           [ 1.,  0.,  1.],
           [ 0.,  1.,  0.]])
    >>> ids
    ['first', 'second', 'third']
    """
    wfull=np.zeros([w.n,w.n],dtype=float)
    keys=w.neighbors.keys()
    if w.id_order:
        keys=w.id_order
    for i,key in enumerate(keys):
        n_i=w.neighbors[key]
        w_i=w.weights[key]
        for j,wij in zip(n_i,w_i):
            c=keys.index(j)
            wfull[i,c]=wij
    return (wfull,keys)

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
        raise Exception, "w must be a spatial weights object"
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
    >>> polyids = get_ids("../examples/columbus.shp", "POLYID")      
    >>> polyids[:5]
    [1, 2, 3, 4, 5]
    """

    try:
        dbname = os.path.splitext(shapefile)[0] + '.dbf'
        db = pysal.open(dbname)
        var = db.by_col[idVariable]
        return var
    except IOError:
        msg = 'The shapefile "%s" appears to be missing its DBF file. The DBF file "%s" could not be found.' % (shapefile, dbname)
        raise IOError, msg
    except AttributeError:
        msg = 'The variable "%s" was not found in the DBF file. The DBF contains the following variables: %s.' % (idVariable, ','.join(db.header))
        raise KeyError, msg

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
    >>> xy = get_points_array_from_shapefile('../examples/juvenile.shp')
    >>> xy[:3]
    array([[ 94.,  93.],
           [ 80.,  95.],
           [ 79.,  90.]])

    Polygon shapefile
    >>> xy = get_points_array_from_shapefile('../examples/columbus.shp')
    >>> xy[:3]
    array([[  8.82721847,  14.36907602],
           [  8.33265837,  14.03162401],
           [  9.01226541,  13.81971908]])
    """

    f = pysal.open(shapefile)
    shapes = f.read()
    if f.type.__name__ == 'Polygon':
        data = np.array([shape.centroid for shape in shapes])
        return data
    elif f.type.__name__ == 'Point':
        data = np.array([shape for shape in shapes])
        return data

def min_threshold_distance(data):
    """
    Get the maximum nearest neighbor distance

    Parameters
    ----------

    data    : array (n,k)
              n observations on k attributes

    Returns
    -------
    nnd    : float
             maximum nearest neighbor distance between the n observations
    
    Examples
    --------
    >>> from pysal.weights.util import min_threshold_distance
    >>> import numpy as np
    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> min_threshold_distance(data)
    1.0

    """

    kd=KDTree(data)
    nn=kd.query(data,k=2,p=2)
    nnd=nn[0].max(axis=0)[1]
    return nnd

def lat2SW(nrows=3,ncols=5,criterion="rook"):
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
    >>> w9=weights.lat2SW(3,3)
    >>> w9[0,1]
    1
    >>> w9[3,6]
    1
    """
    n = nrows*ncols
    diagonals = []
    offsets = []
    if criterion == "rook" or criterion == "queen":
        d = np.ones((1,n))
        for i in range(ncols-1,n,ncols):
            d[0,i] = 0
        diagonals.append(d)
        offsets.append(-1)

        d = np.ones((1,n))
        diagonals.append(d)
        offsets.append(-ncols)

    if criterion == "queen" or criterion == "bishop":
        d = np.ones((1,n))
        for i in range(0,n,ncols):
            d[0,i] = 0
        diagonals.append(d)
        offsets.append(-(ncols-1))

        d = np.ones((1,n))
        for i in range(ncols-1,n,ncols):
            d[0,i] = 0
        diagonals.append(d)
        offsets.append(-(ncols+1))
    data = np.concatenate(diagonals)
    offsets = np.array(offsets)
    m = sparse.dia_matrix((data,offsets),shape=(n,n),dtype=np.int8)
    m = m+m.T
    return m
        
        
def _test():
    """Doc test"""
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
    from pysal import lat2W

    assert (lat2W(5,5).sparse.todense() == lat2SW(5,5).todense()).all()
    assert (lat2W(5,3).sparse.todense() == lat2SW(5,3).todense()).all()
    assert (lat2W(5,3,rook=False).sparse.todense() == lat2SW(5,3,'queen').todense()).all()
    assert (lat2W(50,50,rook=False).sparse.todense() == lat2SW(50,50,'queen').todense()).all()
