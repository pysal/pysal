"""
:mod:`weights` --- Spatial Weights
==================================

"""

__all__ = ['W']
__author__  = "Sergio J. Rey <srey@asu.edu> "
from pysal.common import *
from scipy import sparse, float32
import gc

class W(object):
    """
    Spatial weights

    Parameters
    ----------
    neighbors       : dictionary
                      key is region ID, value is a list of neighbor IDS
                      Example:  {'a':['b'],'b':['a','c'],'c':['b']}
    weights = None  : dictionary
                      key is region ID, value is a list of edge weights
                      If not supplied all edge wegiths are assumed to have a weight of 1.
                      Example: {'a':[0.5],'b':[0.5,1.5],'c':[1.5]}
    id_order = None : list 
                      An ordered list of ids, defines the order of
                      observations when iterating over W if not set,
                      lexicographical ordering is used to iterate and the
                      id_order_set property will return False.  This can be
                      set after creation by setting the 'id_order' property.

    Attributes
    ----------
    asymmetric        : binary
                        True if weights are asymmetric, False if not
    cardinalities     
    diagW2
    diagWtW           
    diagWtW_WW        : array
                        diagonal elements of WW+W'W
    histogram         : list of tuples
                        neighbor histogram (number of neighbors, number of
                        observations with that many neighbors)
    id_order          : list
                        order of observations when iterating over weights
    id_order_set      : binary
                        True if id_order has been set by user, False (default)
    islands           : list
                        ids that have no neighbors
    max_neighbors     : int
                        maximum cardinality 
    min_neighbors     : int 
                        minimum cardinality 
    mean_neighbors    : float
                        average cardinality 
    n                 : int
                        number of observations 
    neighbors         : dictionary (Read Only)
                        {id:[id1,id2]}, key is id, value is list of neighboring
                        ids
    neighbor_offsets  : dictionary
                        like neighbors but the value is a list of zero offset
                        ids
    nonzero           : int
                        number of nonzero weights
    pct_nonzero       : float
                        percentage of all weights that are nonzero
    s0
    s1                
    s2                
    sd                : float
                        standard deviation of number of neighbors 
    sparse
    transform         : string
                        property for weights transformation, can be used to get and set weights transformation 
    transformations   : dictionary
                        transformed weights, key is transformation type, value are weights
    trcWtW            
    trcWtW_WW         : float
                        trace of WW+W'W
    weights           : dictionary (Read Only)
                        key is observation id, value is list of transformed
                        weights in order of neighbor ids (see neighbors)

    Examples
    --------

    >>> neighbors={0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> weights={0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
    >>> w=W(neighbors,weights)
    >>> w.pct_nonzero
    0.29629629629629628

    Read from external gal file

    >>> import pysal
    >>> w=pysal.open("../examples/stl.gal").read()
    >>> w.n
    78
    >>> w.pct_nonzero
    0.065417488494411577

    Set weights implicitly 

    >>> neighbors={0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    >>> w=W(neighbors)
    >>> w.pct_nonzero
    0.29629629629629628
    >>> w=lat2W(100,100)
    >>> w.trcW2
    39600.0
    >>> w.trcWtW
    39600.0
    >>> w.transform='r'
    >>> w.trcW2
    2530.7222222222586
    >>> w.trcWtW
    2533.6666666666774
    >>> 


    Cardinality Histogram

    >>> w=pysal.rook_from_shapefile("../examples/sacramentot2.shp")
    >>> w.histogram
    [(1, 1), (2, 6), (3, 33), (4, 106), (5, 114), (6, 70), (7, 35), (8, 17), (9, 9), (10, 4), (11, 4), (12, 3), (13, 0), (14, 1)]
    >>> 
    
    """
    def __init__(self,neighbors,weights=None,id_order=None):
        """see class docstring"""
        self.transformations={}
        self.neighbors=ROD(neighbors)
        if not weights:
            weights = {}
            for key in neighbors:
                weights[key] = [1.] * len(neighbors[key])
        self.weights=ROD(weights)
        self.transformations['O']=self.weights #original weights
        self.transform='O'
        if id_order == None:
            self._id_order=self.neighbors.keys()
            self._id_order.sort()
            self._id_order_set=False
        else:
            self._id_order=id_order
            self._id_order_set=True
        self._reset()
        self._n=len(self.weights)


    def _reset(self):
        """
        Reset properties
        """
        self._cache={}

    @property
    def sparse(self):
        """
        Sparse representation of weights
        """
        if 'sparse' not in self._cache:
            self._sparse=self._build_sparse()
            self._cache['sparse']=self._sparse
        return self._sparse

    def _build_sparse(self):
        """
        construct the sparse attribute
        """
        
        row=[]
        col=[]
        data=[]
        gc.disable()
        id2i=self.id2i
        for id, neigh_list in self.neighbor_offsets.iteritems():
            card=self.cardinalities[id]
            row.extend([id2i[id]]*card)
            col.extend(neigh_list)
            data.extend(self.weights[id])
        gc.enable()
        row=np.array(row)
        col=np.array(col)
        data=np.array(data)
        s=sparse.csr_matrix((data,(row,col)), shape=(self.n, self.n))
        return s

    @property
    def id2i(self):
        """
        Dictionary where the key is an ID and the value is that ID's
        index in W.id_order.
        """
        if 'id2i' not in self._cache:
            self._id2i={}
            for i,id in enumerate(self._id_order):
                self._id2i[id]=i
            self._id2i=ROD(self._id2i)
            self._cache['id2i']=self._id2i
        return self._id2i

    @property
    def n(self):
        if "n" not in self._cache:
            self._n=len(self.neighbors)
            self._cache['n']=self._n
        return self._n


    @property
    def s0(self):
        """
        float

        .. math::

               s0=\sum_i \sum_j w_{i,j}

        """
        if 's0' not in self._cache:
            self._s0=self.sparse.sum()
            self._cache['s0']=self._s0
        return self._s0

    @property
    def s1(self):
        """
        float

        .. math::

               s1=1/2 \sum_i \sum_j (w_{i,j} + w_{j,i})^2

        """
        if 's1' not in self._cache:
            t=self.sparse.transpose()
            t=t+self.sparse
            t2=t.multiply(t) # element-wise square
            self._s1=t2.sum()/2.
            self._cache['s1']=self._s1
        return self._s1

    @property
    def s2array(self):
        """
        individual elements comprising s2


        See Also
        --------
        s2

        """
        if 's2array' not in self._cache:
            s=self._sparse
            self._s2array= np.array(s.sum(1)+s.sum(0).transpose())**2
            self._cache['s2array']=self._s2array
        return self._s2array

    @property
    def s2(self):
        """
        float


        .. math::

                s2=\sum_j (\sum_i w_{i,j} + \sum_i w_{j,i})^2

        """
        if 's2' not in self._cache:
            self._s2=self.s2array.sum()
            self._cache['s2']=self._s2
        return self._s2

    @property
    def trcW2(self):
        """
        Trace of :math:`WW`

        See Also
        --------
        diagW2

        """
        if 'trcW2' not in self._cache:
            self._trcW2=self.diagW2.sum()
            self._cache['trcw2']=self._trcW2
        return self._trcW2


    @property
    def diagW2(self):
        """
        Diagonal of :math:`WW` : array

        See Also
        --------
        trcW2

        """
        if 'diagw2' not in self._cache:
            self._diagW2=(self.sparse*self.sparse).diagonal()
            self._cache['diagW2']=self._diagW2
        return self._diagW2

    @property
    def diagWtW(self):
        """
        Diagonal of :math:`W^{'}W`  : array

        See Also
        --------
        trcWtW

        """
        if 'diagWtW' not in self._cache:
            self._diagWtW=(self.sparse.transpose()*self.sparse).diagonal()
            self._cache['diagWtW']=self._diagWtW
        return self._diagWtW

    @property 
    def trcWtW(self):
        """
        Trace of :math:`W^{'}W`  : float

        See Also
        --------
        diagWtW

        """
        if 'trcWtW' not in self._cache:
            self._trcWtW=self.diagWtW.sum()
            self._cache['trcWtW']=self._trcWtW
        return self._trcWtW

    @property
    def diagWtW_WW(self):
        if 'diagWtW_WW' not in self._cache:
            wt=self.sparse.transpose()
            w=self.sparse
            self._diagWtW_WW=(wt*w+w*w).diagonal()
            self._cache['diagWtW_WW']=self._diagWtW_WW
        return self._diagWtW_WW

    @property
    def trcWtW_WW(self):
        if 'trcWtW_WW' not in self._cache:
            self._trcWtW_WW=self.diagWtW_WW.sum()
            self._cache['trcWtW_WW']=self._trcWtW_WW
        return self._trcWtW_WW

    @property
    def pct_nonzero(self):
        if 'pct_nonzero' not in self._cache:
            self._pct_nonzero=self.sparse.nnz/(1.*self._n**2)
            self._cache['pct_nonzero']=self._pct_nonzero
        return self._pct_nonzero

    @property
    def cardinalities(self):
        """
        number of neighbors for each observation : dict
        """
        if 'cardinalities' not in self._cache:
            c={}
            for i in self._id_order:
                c[i]=len(self.neighbors[i])
            self._cardinalities=c
            self._cache['cardinalities']=self._cardinalities
        return self._cardinalities

    @property
    def max_neighbors(self):
        if 'max_neighbors' not in self._cache:
            self._max_neighbors=max(self.cardinalities.values())
            self._cache['max_neighbors']=self._max_neighbors
        return self._max_neighbors


    @property
    def mean_neighbors(self):
        if 'max_neighbors' not in self._cache:
            self._mean_neighbors=np.mean(self.cardinalities.values())
            self._cache['max_neighbors']=self._max_neighbors
        return self._mean_neighbors


    @property
    def min_neighbors(self):
        if 'min_neighbors' not in self._cache:
            self._min_neighbors=min(self.cardinalities.values())
            self._cache['min_neighbors']=self._min_neighbors
        return self._min_neighbors


    @property
    def nonzero(self):
        if 'nonzero' not in self._cache:
            self._nonzero=self._sparse.nnz
            self._cache['nonzero']=self._nonzero
        return self._nonzero

    @property
    def sd(self):
        """
        standard deviation of cardinalities : float
        """
        if 'sd' not in self._cache:
            self._sd=np.std(self.cardinalities.values())
            self._cache['sd']=self._sd
        return self._sd


    @property
    def asymmetries(self):
        if 'asymmetries' not in self._cache:
            self._asymmetries=self.asymmetry()
            self._cache['asymmetries']=self._asymmetries
        return self._asymmetries

    @property
    def islands(self):
        if 'islands' not in self._cache:
            self._islands = [i for i,c in self.cardinalities.items() if c==0]
            self._cache['islands']=self._islands
        return self._islands


    @property
    def histogram(self):
        if 'histogram' not in self._cache:
            ct,bin=np.histogram(self.cardinalities.values(),range(self.min_neighbors,self.max_neighbors+2))
            self._histogram=zip(bin,ct)
            self._cache['histogram']=self._histogram
        return self._histogram


    def __getitem__(self,key):
        """
        Allow a dictionary like interaction with the weights class.

        Examples
        --------
        >>> from Contiguity import buildContiguity
        >>> w=buildContiguity(pysal.open('../examples/10740.shp'),criterion='rook')
        >>> w[0]
        {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
        >>> w = lat2W()
        >>> w[1]
        {0: 1.0, 2: 1.0, 6: 1.0}
        >>> w[0]
        {1: 1.0, 5: 1.0}
        """
        return dict(zip(self.neighbors[key],self.weights[key]))


    def __iter__(self):
        """
        Support iteration over weights

        Examples
        --------
        >>> w=pysal.lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print i,wi
        ...     
        0 {1: 1.0, 3: 1.0}
        1 {0: 1.0, 2: 1.0, 4: 1.0}
        2 {1: 1.0, 5: 1.0}
        3 {0: 1.0, 4: 1.0, 6: 1.0}
        4 {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0}
        5 {8: 1.0, 2: 1.0, 4: 1.0}
        6 {3: 1.0, 7: 1.0}
        7 {8: 1.0, 4: 1.0, 6: 1.0}
        8 {5: 1.0, 7: 1.0}
        >>> 
        """
        class _W_iter:
            def __init__(self,w):
                self.w = w
                self.n = len(w._id_order)
                self._idx = 0
            def next(self):
                if self._idx >= self.n:
                    self._idx=0
                    raise StopIteration
                value = self.w.__getitem__(self.w._id_order[self._idx])
                self._idx+=1
                return value
        return _W_iter(self)

    def __set_id_order(self, ordered_ids):
        """
        Set the iteration order in w.

        W can be iterated over. On construction the iteration order is set to
        the lexicographic order of the keys in the w.weights dictionary. If a specific order
        is required it can be set with this method.

        Parameters
        ----------

        ordered_ids : sequence
                      identifiers for observations in specified order

        Notes
        -----

        ordered_ids is checked against the ids implied by the keys in
        w.weights. If they are not equivalent sets an exception is raised and
        the iteration order is not changed.

        Examples
        --------

        >>> w=lat2W(3,3)
        >>> for i,wi in enumerate(w):
        ...     print i,wi
        ...     
        0 {1: 1.0, 3: 1.0}
        1 {0: 1.0, 2: 1.0, 4: 1.0}
        2 {1: 1.0, 5: 1.0}
        3 {0: 1.0, 4: 1.0, 6: 1.0}
        4 {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0}
        5 {8: 1.0, 2: 1.0, 4: 1.0}
        6 {3: 1.0, 7: 1.0}
        7 {8: 1.0, 4: 1.0, 6: 1.0}
        8 {5: 1.0, 7: 1.0}

        >>> w.id_order
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> w.id_order=range(8,-1,-1)
        >>> w.id_order
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> for i,w_i in enumerate(w):
        ...     print i,w_i
        ...     
        0 {5: 1.0, 7: 1.0}
        1 {8: 1.0, 4: 1.0, 6: 1.0}
        2 {3: 1.0, 7: 1.0}
        3 {8: 1.0, 2: 1.0, 4: 1.0}
        4 {1: 1.0, 3: 1.0, 5: 1.0, 7: 1.0}
        5 {0: 1.0, 4: 1.0, 6: 1.0}
        6 {1: 1.0, 5: 1.0}
        7 {0: 1.0, 2: 1.0, 4: 1.0}
        8 {1: 1.0, 3: 1.0}
        >>> 
        
        """
        if set(self._id_order) == set(ordered_ids):
            self._id_order=ordered_ids
            self._idx=0
            self._id_order_set=True
            self._reset()
        else:
            raise Exception, 'ordered_ids do not align with W ids'

    def __get_id_order(self):
        """returns the ids for the observations in the order in which they
        would be encountered if iterating over the weights."""
        return self._id_order

    id_order=property(__get_id_order, __set_id_order)

    @property
    def id_order_set(self):
        """returns True if user has set id_order, False if not.

        Example
        >>> w=lat2W()
        >>> w.id_order_set
        True
        """
        return self._id_order_set


    @property
    def neighbor_offsets(self):
        """
        Given the current id_order, neighbor_offsets[id] is the offsets of the
        id's neighrbors in id_order

        Examples
        --------

        >>> neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
        >>> weights ={'c': [1.0], 'b': [1.0, 1.0], 'a': [1.0]}
        >>> w=W(neighbors,weights)
        >>> w.id_order = ['a','b','c']
        >>> w.neighbor_offsets['b']
        [2, 0]
        >>> w.id_order = ['b','a','c']
        >>> w.neighbor_offsets['b']
        [2, 1]
        """
        if "neighbors_0" not in self._cache:
            self.__neighbors_0={}
            id2i=self.id2i
            for id, neigh_list in self.neighbors.iteritems():
                self.__neighbors_0[id]=[id2i[neigh] for neigh in neigh_list] 
            self._cache['neighbors_0']=self.__neighbors_0
        return self.__neighbors_0


    def get_transform(self):
        """
        Getter for transform property

        Returns
        -------
        transformation : string (or none)

        Examples
        --------
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]
        >>> 
        """
        return self._transform

    def set_transform(self, value="B"):
        """
        Transformations of weights.

        Parameters
        ----------
        transform : string (not case sensitive)
                    B: Binary 
                    R: Row-standardization (global sum=n)
                    D: Double-standardization (global sum=1)
                    V: Variance stabilizing
                    O: Restore original transformation (from instantiation)

        Examples
        --------
        >>> w=lat2W()
        >>> w.weights[0]
        [1.0, 1.0]
        >>> w.transform
        'O'
        >>> w.transform='r'
        >>> w.weights[0]
        [0.5, 0.5]
        >>> w.transform='b'
        >>> w.weights[0]
        [1.0, 1.0]
        >>> 
        """
        value=value.upper()
        self._transform = value
        if self.transformations.has_key(value):
            self.weights=self.transformations[value]
            self._reset()
        else:
            if value == "R": 
                # row standardized weights
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    row_sum=sum(wijs)*1.0
                    weights[i]=[wij/row_sum for wij in wijs]
                weights=ROD(weights)
                self.transformations[value]=weights
                self.weights=weights
                self._reset()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self._reset()
                s0=self.s0
                ws=1.0/s0
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[wij*ws for wij in wijs]
                weights=ROD(weights)
                self.transformations[value]=weights
                self.weights=weights
                self._reset()
            elif value == "B":
                # binary transformation
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[1.0 for wij in wijs]
                weights=ROD(weights)
                self.transformations[value]=weights
                self.weights=weights
                self._reset()
            elif value == "V":
                # variance stabilizing
                weights={}
                q={}
                k=self.cardinalities
                s={}
                Q=0.0
                for i in self.weights:
                    wijs = self.weights[i]
                    q[i] = math.sqrt(sum([wij*wij for wij in wijs]))
                    s[i] = [wij / q[i] for wij in wijs]
                    Q+=sum([si for si in s[i]])
                nQ=self.n/Q
                for i in self.weights:
                    weights[i] = [ w*nQ for w in s[i]]
                weights=ROD(weights)
                self.transformations[value]=weights
                self.weights=weights
                self._reset()
            elif value =="O":
                # put weights back to original transformation
                weights={}
                original=self.transformations[value]
                self.weights=original
                self._reset()
            else:
                print 'unsupported weights transformation'

    transform = property(get_transform, set_transform)
    

    def asymmetry(self):
        """
        Checks for w_{i,j} == w_{j,i} forall i,j

        Returns
        -------
        asymmetries : list 
                      empty if no asymmetries are found
                      if asymmetries, first list is row indices, second
                      list is column indices of asymmetric cells

        Examples
        --------

        >>> w=lat2W(3,3)
        >>> w.asymmetry()
        []
        >>> w.transform='r'
        >>> w.asymmetry()
        (array([1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 5,
               7]), array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8,
               8]))
        >>> neighbors={0:[1,2,3], 1:[1,2,3], 2:[0,1], 3:[0,1]}
        >>> weights={0:[1,1,1], 1:[1,1,1], 2:[1,1], 3:[1,1]}
        >>> w=W(neighbors,weights)
        >>> w.asymmetry()
        (array([1, 0]), array([0, 1]))

        """

        wd=self.sparse.transpose()-self.sparse
        ids=np.nonzero(wd)
        if len(ids[0])==0:
            return []
        else:
            return ids


    def full(self):
        """
        Generate a full numpy array

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
        >>> wf,ids=w.full()
        >>> wf
        array([[ 0.,  1.,  0.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  0.]])
        >>> ids
        ['first', 'second', 'third']

        See also
        --------
        full
        """
        return full(self)


    def shimbel(self):
        """
        Find the Shmibel matrix for the first order contiguity matrix.
        
        Returns
        -------

        implicit : list of lists
                   one list for each observation which stores the shortest
                   order between it and each of the the other observations.

        Examples
        --------
        >>> w5=lat2W()
        >>> w5_shimbel=w5.shimbel()
        >>> w5_shimbel[0][24]
        8
        >>> w5_shimbel[0][0:4]
        [-1, 1, 2, 3]
        >>>

        See Also
        --------
        shimbel

        """
        return shimbel(self)


    def order(self,kmax=3):
        """
        Determine the non-redundant order of contiguity up to a specific
        order.

        Parameters
        ----------

        kmax    : int
                  maximum order of contiguity

        Returns
        -------

        implicit : dict
                   observation id is the key, value is a list of contiguity
                   orders with a negative 1 in the ith position


        Notes
        -----
        Implements the algorithm in Anselin and Smirnov (1996) [1]_


        Examples
        --------
        >>> from Contiguity import buildContiguity
        >>> w=buildContiguity(pysal.open('../examples/10740.shp'),criterion='rook')
        >>> w3=w.order()
        >>> w3[1][0:5]
        [1, -1, 1, 2, 1]

        References
        ----------
        .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
           constructing proper higher order spatial lag operators. Journal of
           Regional Science, 36, 67-89. 

        See also
        --------
        order

        """
        return order(self,kmax)


    def higher_order(self,k=3):
        """
        Contiguity weights object of order k 

        Parameters
        ----------

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
        >>> w5=lat2W()
        >>> w5_shimbel=w5.shimbel()
        >>> w5_shimbel[0][24]
        8
        >>> w5_shimbel[0][0:4]
        [-1, 1, 2, 3]
        >>> w5_8th_order=w5.higher_order(8)
        >>> w5_8th_order.neighbors[0]
        [24]
        >>> from Contiguity import buildContiguity
        >>> w=buildContiguity(pysal.open('../examples/10740.shp'),criterion='rook')
        >>> w2=w.higher_order(2)
        >>> w[1]
        {0: 1.0, 2: 1.0, 83: 1.0, 4: 1.0}
        >>> w2[1]
        {3: 1.0, 5: 1.0, 6: 1.0, 10: 1.0, 82: 1.0, 85: 1.0, 91: 1.0, 92: 1.0, 101: 1.0}
        >>> w[147]
        {144: 1.0, 146: 1.0, 164: 1.0, 165: 1.0, 150: 1.0}
        >>> w[85]
        {0: 1.0, 101: 1.0, 83: 1.0, 84: 1.0, 90: 1.0, 91: 1.0, 93: 1.0}
        >>> 

        References
        ----------
        .. [1] Anselin, L. and O. Smirnov (1996) "Efficient algorithms for
           constructing proper higher order spatial lag operators. Journal of
           Regional Science, 36, 67-89. 

        See also
        --------
        higher_order
        """
        return higher_order(self,k)


from util import *
import util
__all__ += util.__all__
from Distance import *
from Contiguity import *
from user import *
from spatial_lag import *
import Wsets


if __name__ == "__main__":

    import doctest
    doctest.testmod()
