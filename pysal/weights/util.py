from pysal.common import *
from pysal.weights import W

__all__ = ['lat2W','regime_weights','comb','order', 'higher_order', 'shimbel', 'full']

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
              type of matrix. Default is rook. For queen, rook =False
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

    >>> w9=lat2W(3,3)
    >>> w9.pct_nonzero
    0.29629629629629628
    >>> w9[0]
    {1: 1.0, 3: 1.0}
    >>> w9[3]
    {0: 1.0, 4: 1.0, 6: 1.0}
    >>> 
    """

    n=nrows*ncols
    r1=nrows-1
    c1=ncols-1
    rid=[ i/ncols for i in xrange(n) ]
    cid=[ i%ncols for i in xrange(n) ]
    w={}
    
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
    return W(w,weights,ids)

def regime_weights(regimes):
    """
    Construct spatial weights for regime neighbors.

    Block contiguity structures are relevant when defining neighbor relations
    based on membership in a regime. For example, all counties belonging to
    the same state could be defined as neighbors.

    Parameters
    ----------
    regimes : list or array
           ids of which regime an observation belongs to

    Returns
    -------

    W : spatial weights instance

    Examples
    --------
    
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
    region_ids=list(set(regimes))
    regime=np.array(regimes)
    neighbors={}
    weights={}
    ids=np.arange(len(regimes))
    regions=[ids[regime==region] for region in region_ids]
    n=len(regimes)
    for i in xrange(n):
        neighbors[i]=[]
    for region in regions:
        for i,j in comb(region.tolist(),2):
            neighbors[i].append(j)
            neighbors[j].append(i)
    weights={}
    for i,nn in neighbors.items():
        weights[i]=[1.]*len(nn)
    return W(neighbors,weights)

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


    Example
    -------
    >>> from Contiguity import buildContiguity
    >>> w=buildContiguity('../examples/10740.shp',criterion='rook')
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
    return W(neighbors,weights)

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
    >>> w5=lat2W()
    >>> w5_shimbel=shimbel(w5)
    >>> w5_shimbel[0][24]
    8
    >>> w5_shimbel[0][0:4]
    [-1, 1, 2, 3]
    >>>
    """
    info={}
    ids=w.neighbors.keys()
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
                next_neighbors=w.neighbors[p]
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

    >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
    >>> weights={'first':[1],'second':[1,1],'third':[1]}
    >>> w=W(neighbors,weights)
    >>> wf,ids=full(w)
    >>> wf
    array([[ 0.,  1.,  1.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.]])
    >>> ids
    ['second', 'third', 'first']
    """
    wfull=np.zeros([w.n,w.n],dtype=float)
    keys=w.neighbors.keys()
    for i,key in enumerate(keys):
        n_i=w.neighbors[key]
        w_i=w.weights[key]
        for j,wij in zip(n_i,w_i):
            c=keys.index(j)
            wfull[i,c]=wij
    return (wfull,keys)
   
if __name__ == "__main__":

    import doctest
    doctest.testmod()
