"""
Distance based spatial weights
"""

__author__  = "Sergio J. Rey <srey@asu.edu> "

import pysal
import scipy.spatial
from pysal.common import *
from pysal.weights import W

from scipy import sparse

__all__ = ["knnW", "Kernel", "DistanceBand"]

def knnW(data,k=2,p=2,ids=None):
    """
    Creates nearest neighbor weights matrix based on k nearest 
    neighbors.
    
    Parameters
    ----------

    data       : array (n,k) or KDTree where KDtree.data is array (n,k)
                 n observations on k characteristics used to measure
                 distances between the n objects
    k          : int
                 number of nearest neighbors
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    ids        : list
                 identifiers to attach to each observation
    Returns
    -------

    w         : W instance
                Weights object with binary weights


    Examples
    --------

    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wnn2=knnW(data,k=2)
    >>> wnn4=knnW(data,k=4)
    >>> wnn4.neighbors[0]
    [1, 5, 6, 2]
    >>> wnn4.neighbors[5]
    [0, 6, 10, 1]
    >>> wnn2.neighbors[0]
    [1, 5]
    >>> wnn2.neighbors[5]
    [0, 6]
    >>> wnn2.pct_nonzero
    0.080000000000000002
    >>> wnn4.pct_nonzero
    0.16
    >>> wnn3e=knnW(data,p=2,k=3)
    >>> wnn3e.neighbors[0]
    [1, 5, 6]
    >>> wnn3m=knnW(data,p=1,k=3)
    >>> wnn3m.neighbors[0]
    [1, 5, 2]


    Notes
    -----

    Ties between neighbors of equal distance are arbitrarily broken.

    See Also
    --------
    pysal.weights.W
    """
    # handle data
    if issubclass(type(data),scipy.spatial.KDTree):
        kd = data
        data = kd.data
    elif type(data).__name__=='ndarray':
        kd=KDTree(data)
    else:
        print 'Unsupported  type'

    # calculate
    nnq=kd.query(data,k=k+1,p=p)
    info=nnq[1]
    neighbors={}
    weights={}
    if ids:
        idset = np.array(ids)
    else:
        idset = np.arange(len(info))
    for i, row in enumerate(info):
        row = row.tolist()
        row.remove(i)
        neighbors[idset[i]] = list(idset[row])
        weights[idset[i]] = [1]*len(neighbors[idset[i]])

    return pysal.weights.W(neighbors,weights=weights,id_order=ids)


class Kernel(W):
    """Spatial weights based on kernel functions
    
    
    Parameters
    ----------

    data        : array (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    bandwidth   : float or array-like (optional)
                  the bandwidth :math:`h_i` for the kernel. 
    fixed       : binary
                  If true then :math:`h_i=h \\forall i`. If false then
                  bandwidth is adaptive across observations.
    k           : int
                  the number of nearest neighbors to use for determining
                  bandwidth. For fixed bandwidth, :math:`h_i=max(dknn) \\forall i`
                  where :math:`dknn` is a vector of k-nearest neighbor
                  distances (the distance to the kth nearest neighbor for each
                  observation).  For adaptive bandwidths, :math:`h_i=dknn_i`
    function    : string {'triangular','uniform','quadratic','quartic','gaussian'}
                  kernel function defined as follows with 

                  .. math::

                      z_{i,j} = d_{i,j}/h_i

                  triangular 

                  .. math::

                      K(z) = (1 - |z|) \ if |z| \le 1

                  uniform 

                  .. math::

                      K(z) = 1/2 \ if |z| \le 1

                  quadratic 

                  .. math::

                      K(z) = (3/4)(1-z^2) \ if |z| \le 1


                  quartic

                  .. math::

                      K(z) = (15/16)(1-z^2)^2 \ if |z| \le 1
                 

                  gaussian

                  .. math::

                      K(z) = (2\pi)^{(-1/2)} exp(-z^2 / 2)

    eps         : float
                  adjustment to ensure knn distance range is closed on the
                  knnth observations

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> kw=Kernel(points)
    >>> kw.weights[0]
    [1.0, 0.500000049999995, 0.4409830615267465]
    >>> kw.neighbors[0]
    [0, 1, 3]
    >>> kw.bandwidth
    array([[ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002],
           [ 20.000002]])
    >>> kw15=Kernel(points,bandwidth=15.0)
    >>> kw15[0]
    {0: 1.0, 1: 0.33333333333333337, 3: 0.2546440075000701}
    >>> kw15.neighbors[0]
    [0, 1, 3]
    >>> kw15.bandwidth
    array([[ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.],
           [ 15.]])

    Adaptive bandwidths user specified

    >>> bw=[25.0,15.0,25.0,16.0,14.5,25.0]
    >>> kwa=Kernel(points,bandwidth=bw)
    >>> kwa.weights[0]
    [1.0, 0.6, 0.552786404500042, 0.10557280900008403]
    >>> kwa.neighbors[0]
    [0, 1, 3, 4]
    >>> kwa.bandwidth
    array([[ 25. ],
           [ 15. ],
           [ 25. ],
           [ 16. ],
           [ 14.5],
           [ 25. ]])

    Endogenous adaptive bandwidths 

    >>> kwea=Kernel(points,fixed=False)
    >>> kwea.weights[0]
    [1.0, 0.10557289844279438, 9.99999900663795e-08]
    >>> kwea.neighbors[0]
    [0, 1, 3]
    >>> kwea.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])

    Endogenous adaptive bandwidths with Gaussian kernel

    >>> kweag=Kernel(points,fixed=False,function='gaussian')
    >>> kweag.weights[0]
    [0.3989422804014327, 0.2674190291577696, 0.2419707487162134]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])
    """
    def __init__(self,data,bandwidth=None,fixed=True,k=2,
                 function='triangular',eps=1.0000001,ids=None):
        if issubclass(type(data),scipy.spatial.KDTree):
            self.kdt = data
            self.data = self.kdt.data
            data = self.data
        else:
            self.data=data
            self.kdt=KDTree(self.data)
        self.k=k+1 
        self.function=function.lower()
        self.fixed=fixed
        self.eps=eps
        if bandwidth:
            try:
                bandwidth=np.array(bandwidth)
                bandwidth.shape=(len(bandwidth),1)
            except:
                bandwidth=np.ones((len(data),1),'float')*bandwidth
            self.bandwidth=bandwidth
        else:
            self._set_bw()

        self._eval_kernel()
        neighbors, weights = self._k_to_W(ids)
        W.__init__(self,neighbors,weights,ids)

    def _k_to_W(self, ids=None):
        allneighbors={}
        weights={}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self.data))
        for i, neighbors in enumerate(self.kernel):
            if len(self.neigh[i]) == 0:
                allneighbors[ids[i]] = []
                weights[ids[i]] = []
            else:    
                allneighbors[ids[i]] = list(ids[self.neigh[i]])
                weights[ids[i]] = self.kernel[i].tolist()
        return allneighbors,weights

    def _set_bw(self):
        dmat,neigh=self.kdt.query(self.data,k=self.k)
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth=dmat.max()*self.eps
            n=len(dmat)
            self.bandwidth=np.ones((n,1),'float')*bandwidth
        else:
            # use local max knn distance
            self.bandwidth=dmat.max(axis=1)*self.eps
            self.bandwidth.shape=(self.bandwidth.size,1)
            # identify knn neighbors for each point
            nnq=self.kdt.query(self.data,k=self.k)
            self.neigh=nnq[1]

    def _eval_kernel(self):
        # get points within bandwidth distance of each point
        if not hasattr(self, 'neigh'):
            kdtq=self.kdt.query_ball_point
            neighbors=[kdtq(self.data[i],r=bwi[0]) for i,bwi in enumerate(self.bandwidth)]
            self.neigh=neighbors
        # get distances for neighbors
        data=np.array(self.data)
        bw=self.bandwidth
        z=[]
        for i,nids in enumerate(self.neigh):
            di=data[np.array([0,i])]
            ni=data[nids]
            zi=cdist(di,ni)[1]/bw[i]
            z.append(zi)
        zs=z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function=='triangular':
            self.kernel=[1-z for z in zs]
        elif self.function=='uniform':
            self.kernel=[np.ones(z.shape)*0.5 for z in zs]
        elif self.function=='quadratic':
            self.kernel=[(3./4)*(1-z**2) for z in zs]
        elif self.function=='quartic':
            self.kernel=[(15./16)*(1-z**2)**2 for z in zs]
        elif self.function=='gaussian':
            c=np.pi*2
            c=c**(-0.5)
            self.kernel=[c*np.exp(-(z**2)/2.) for z in zs]
        else:
            print 'Unsupported kernel function',self.function
        

class DistanceBand(W):
    """Spatial weights based on distance band

    Parameters
    ----------

    data        : array (n,k) or KDTree where KDtree.data is array (n,k)
                  n observations on k characteristics used to measure
                  distances between the n objects
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance
    binary     : binary
                 If true w_{ij}=1 if d_{i,j}<=threshold, otherwise w_{i,j}=0
                 If false wij=dij^{alpha}
    alpha      : float 
                 distance decay parameter for weight (default -1.0)
                 if alpha is positive the weights will not decline with
                 distance. If binary is True, alpha is ignored 

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> w=DistanceBand(points,threshold=11.2)
    >>> w.weights
    {0: [1, 1], 1: [1, 1], 2: [], 3: [1, 1], 4: [1], 5: [1]}
    >>> w.neighbors
    {0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]}

    inverse distance weights 

    >>> w=DistanceBand(points,threshold=11.2,binary=False)
    >>> w.weights[0]
    [0.10000000000000001, 0.089442719099991588]
    >>> w.neighbors[0]
    [1, 3]
    >>> 

    gravity weights

    >>> w=DistanceBand(points,threshold=11.2,binary=False,alpha=-2.)
    >>> w.weights[0]
    [0.01, 0.0079999999999999984]


    Notes
    -----

    this was initially implemented running scipy 0.8.0dev (in epd 6.1).
    earlier versions of scipy (0.7.0) have a logic bug in scipy/sparse/dok.py
    so serge changed line 221 of that file on sal-dev to fix the logic bug

    """
    def __init__(self,data,threshold,p=2,alpha=-1.0,binary=True,ids=None):
        """
        Casting to floats is a work around for a bug in scipy.spatial.  See detail in pysal issue #126
        """
        if issubclass(type(data),scipy.spatial.KDTree):
            self.kd = data
            self.data = self.kd.data
        else:
            try:
                data = np.asarray(data)
                if data.dtype.kind != 'f':
                    data = data.astype(float)
                self.data = data
                self.kd=KDTree(self.data)
            except:
                raise ValueError, "Could not make array from data"

        self.p=p
        self.threshold=threshold
        self.binary=binary
        self.alpha=alpha
        self._band()
        neighbors, weights = self._distance_to_W(ids)
        W.__init__(self,neighbors,weights,ids)

    def _band(self):
        """ 
        find all pairs within threshold
        """
        kd=self.kd
        #ns=[kd.query_ball_point(point,self.threshold) for point in self.data]
        ns=kd.query_ball_tree(kd,self.threshold)
        self._nmat=ns

    def _distance_to_W(self,ids=None):
        allneighbors={}
        weights={}
        if ids:
            ids = np.array(ids)
        else:
            ids = np.arange(len(self._nmat))
        if self.binary:
            for i,neighbors in enumerate(self._nmat):
                ns=[ni for ni in neighbors if ni!=i]
                neigh = list(ids[ns])
                if len(neigh) == 0:
                    allneighbors[ids[i]] = []
                    weights[ids[i]] = []
                else:        
                    allneighbors[ids[i]] = neigh
                    weights[ids[i]] = [1]*len(ns)
        else:
            self.dmat=self.kd.sparse_distance_matrix(self.kd,max_distance=self.threshold)
            for i,neighbors in enumerate(self._nmat):
                ns=[ni for ni in neighbors if ni!=i]
                neigh = list(ids[ns])
                if len(neigh) == 0:
                    allneighbors[ids[i]] = []
                    weights[ids[i]] = []
                else:        
                    allneighbors[ids[i]] = neigh
                    weights[ids[i]] = [self.dmat[(i,j)]**self.alpha for j in ns] 
        return allneighbors,weights

def _test():
    """Doc test"""
    import doctest
    doctest.testmod(verbose=False)

if __name__ == "__main__":
    _test()
    
    x=np.arange(1,5)

    n=len(x)
    data=[]
    row=[]
    col=[]
    rdata=[]

    for i in range(n-1):
        data.extend(x[i+1:])
        
        rdata.extend((x[i]*np.ones((1,n-i-1))).tolist()[0]
)
        row.extend([i]*(n-i-1))
        col.extend(range(i+1,n))
    L=sparse.csr_matrix((data,(row,col)),shape=(n,n))
    R=sparse.csr_matrix((rdata,(row,col)),shape=(n,n))
    dlr=L-R
    D=dlr.multiply(dlr)
