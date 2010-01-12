"""
Distance Based Spatial Weights for PySAL

All the classes here extend the spatial :class:`pysal.weights.weights.W` class by defining the weights based
on various distance functions. See :mod:`pysal.weights.weights` for further
information.
"""

__author__  = "Sergio J. Rey <srey@asu.edu> "


from pysal.weights import W
from pysal.common import *

class InverseDistance(W):
    """Creates spatial weights based on inverse distance
    
    Parameters
    ----------

    data            : array (n,k)
                      attribute data, n observations on m attributes
    p               : float
                      Minkowski p-norm distance metric parameter:
                      1<=p<=infinity
                      2: Euclidean distance
                      1: Manhattan distance
    row_standardize : binary
                      If true weights are row standardized, if false weights
                      are not standardized

   
    Examples
    --------

    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wid=InverseDistance(data)
    >>> wid_ns=InverseDistance(data,row_standardize=False)
    >>> wid.weights[0][0:3]
    [0.0, 0.21689522769159933, 0.054223806922899832]
    >>> wid_ns.weights[0][0:3]
    [0.0, 1.0, 0.25]
    """
    def __init__(self,data,p=2,row_standardize=True):
        self.data=data
        self.p=p
        self._distance()
        W.__init__(self,self._distance_to_W())
        if row_standardize:
            self.transform="r"


    def _distance(self):
        dmat=distance_matrix(self.data,self.data,self.p)
        n,k=dmat.shape
        imat=np.identity(n)
        self.dmat=(dmat+imat)**(-self.p) - imat
        self.n=n

    def _distance_to_W(self):
        neighbors={}
        weights={}
        ids=np.arange(self.n)
        
        for i,row in enumerate(self.dmat):
            weights[i]=row.tolist()
            neighbors[i]=np.nonzero(ids!=0)[0].tolist()

        return {"neighbors":neighbors,"weights":weights}
        


class NearestNeighbors(W):
    """Creates contiguity matrix based on k nearest neighbors
    
    Parameters
    ----------

    data       : array (n,k)
                 attribute data, n observations on m attributes
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance

    Examples
    --------

    >>> x,y=np.indices((5,5))
    >>> x.shape=(25,1)
    >>> y.shape=(25,1)
    >>> data=np.hstack([x,y])
    >>> wnn2=NearestNeighbors(data,k=2)
    >>> wnn4=NearestNeighbors(data,k=4)
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
    """
    def __init__(self,data,k=2,p=2):
        self.data=data
        self.k=k
        self.p=p
        self._distance()
        W.__init__(self,self._distance_to_W())


    def _distance(self):
        kd=KDTree(self.data)
        nnq=kd.query(self.data,k=self.k+1,p=self.p)
        self.dmat=nnq

    def _distance_to_W(self):
        info=self.dmat[1]
        neighbors={}
        weights={}
        for row in info:
            i=row[0]
            neighbors[i]=row[1:].tolist()
            weights[i]=[1]*len(neighbors[i])
        return {"neighbors":neighbors,"weights":weights}

class DistanceBand(W):
    """Creates contiguity matrix based on distance band

    Parameters
    ----------

    data       : array (n,k)
                 attribute data, n observations on m attributes
    threshold  : float
                 distance band
    p          : float
                 Minkowski p-norm distance metric parameter:
                 1<=p<=infinity
                 2: Euclidean distance
                 1: Manhattan distance

    Examples
    --------

    >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
    >>> w=DistanceBand(points,threshold=11.2)
    >>> w.weights
    {0: [1, 1], 1: [1, 1], 2: [], 3: [1, 1], 4: [1], 5: [1]}
    >>> w.neighbors
    {0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]}
    >>> 
    """
    def __init__(self,data,threshold,p=2):
        
        self.data=data
        self.p=p
        self.threshold=threshold
        self._distance()
        W.__init__(self,self._distance_to_W())


    def _distance(self):
        kd=KDTree(self.data)
        ns=[kd.query_ball_point(point,self.threshold) for point in self.data]
        self.dmat=ns

    def _distance_to_W(self):
        allneighbors={}
        weights={}
        for i,neighbors in enumerate(self.dmat):
            ns=[ni for ni in neighbors if ni!=i]
            allneighbors[i]=ns
            weights[i]=[1]*len(ns)
        return {"neighbors":allneighbors,"weights":weights}


class Kernel(W):
    """Spatial weights based on kernel functions
    
    
    Parameters
    ----------

    data        : array (n,k)
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

                      K(z) = |z| \ if |z| \le 1

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
    [1.0, 0.50000004999999503, 0.44098306152674649]
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
    [1.0, 0.59999999999999998, 0.55278640450004202, 0.10557280900008403]
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
    [1.0, 0.10557289844279438, 9.9999990066379496e-08]
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
    [0.3989422804014327, 0.26741902915776961, 0.24197074871621341]
    >>> kweag.bandwidth
    array([[ 11.18034101],
           [ 11.18034101],
           [ 20.000002  ],
           [ 11.18034101],
           [ 14.14213704],
           [ 18.02775818]])
    """
    def __init__(self,data,bandwidth=None,fixed=True,k=2,
                 function='triangular',eps=1.0000001):
        self.data=data
        self.k=k+1 
        self.function=function.lower()
        self.fixed=fixed
        self.eps=eps
        self.kdt=KDTree(self.data)
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
        W.__init__(self,self._k_to_W())

    def _k_to_W(self):
        allneighbors={}
        weights={}
        for i,neighbors in enumerate(self.kernel):
            allneighbors[i]=self.neigh[i]
            weights[i]=self.kernel[i].tolist()
        return {"neighbors":allneighbors,"weights":weights}

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

    def _eval_kernel(self):
        # get points within bandwidth distance of each point
        kdtq=self.kdt.query_ball_point
        neighbors=[kdtq(self.data,r=bwi[0])[i] for i,bwi in enumerate(self.bandwidth)]
        self.neigh=neighbors
        # get distances for neighbors
        data=np.array(self.data)
        bw=self.bandwidth
        z=[]
        for i,nids in enumerate(neighbors):
            di=data[np.array([0,i])]
            ni=data[nids]
            zi=cdist(di,ni)[1]/bw[i]
            z.append(zi)
        zs=z
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function=='triangular':
            self.kernel=[1-z for z in zs]
        elif self.function=='uniform':
            self.kernel=z
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
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()

