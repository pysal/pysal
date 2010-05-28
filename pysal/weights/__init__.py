"""
:mod:`weights` --- Spatial Weights
==================================

"""

__all__ = ['W']
__author__  = "Sergio J. Rey <srey@asu.edu> "
from pysal.common import *
from scipy import sparse, float32
import gc

# constant for precision
DELTA = 0.0000001

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
    cardinalities     : dictionary 
                        number of neighbors for each observation 
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
    neighbors         : dictionary
                        {id:[id1,id2]}, key is id, value is list of neighboring
                        ids
    neighbor_offsets  : dictionary
                        like neighbors but the value is a list of zero offset
                        ids
    nonzero           : int
                        number of nonzero weights
    pct_nonzero       : float
                        percentage of all weights that are nonzero
    s0                : float
                        sum of all weights 
    s1                : float
                        trace of ww
    s2                : float
                        trace of w'w
    sd                : float
                        standard deviation of number of neighbors 
    transform         : string
                        property for weights transformation, can be used to get and set weights transformation 
    transformations   : dictionary
                        transformed weights, key is transformation type, value are weights
    weights           : dictionary
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


    """
    def __init__(self,neighbors,weights=None,id_order=None):
        """see class docstring"""
        self.transformations={}
        self.neighbors=neighbors
        if not weights:
            weights = {}
            for key in neighbors:
                weights[key] = [1.] * len(neighbors[key])
        self.weights=weights
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
        self._sparse=self._build_sparse()
        self._sparse_set=True

    def _reset(self):
        """
        Reset properties
        """
        self.__neighbors_0 = False
        self._id2i=None
        self._idx=0
        self._n=None
        self._s0=None
        self._s1=None
        self._s2=None
        self._pct_nonzero=None
        self._cardinalities=None
        self._max_neighbors=None
        self._mean_neighbors=None
        self._min_neighbors=None
        self._nonzero=None
        self._sd=None
        self._asymmetries=None
        self._islands=None
        self._histogram=None
        self._sparse_set=False
        self._sparse=None

    @property
    def sparse(self):
        if not self._sparse_set:
            self._sparse=self._build_sparse()
            self._sparse_set=True
        return self._sparse

    def _build_sparse(self):
        """
        localizing this for optimization
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
        """Dictionary where the key is an ID and the value is that ID's
        index in W.id_order."""
        if not self._id2i:
            self._id2i={}
            for i,id in enumerate(self._id_order):
                self._id2i[id]=i
        return self._id2i

    @property
    def n(self):
        self._n=len(self.neighbors)
        return self._n

    @property
    def s0(self):
        if not self._s0:
            self.characteristics()
        return self._s0

    @property
    def s1(self):
        if not self._s1:
            self.characteristics()
        return self._s1

    @property
    def s2(self):
        if not self._s2:
            self.characteristics()
        return self._s2
    @property
    def pct_nonzero(self):
        if not self._pct_nonzero:
            self.characteristics()
        return self._pct_nonzero

    @property
    def cardinalities(self):
        if not self._cardinalities:
            self.characteristics()
        return self._cardinalities

    @property
    def max_neighbors(self):
        if not self._max_neighbors:
            self.characteristics()
        return self._max_neighbors


    @property
    def mean_neighbors(self):
        if not self._mean_neighbors:
            self.characteristics()
        return self._mean_neighbors


    @property
    def mean_neighbors(self):
        if not self._mean_neighbors:
            self.characteristics()
        return self._mean_neighbors

    @property
    def min_neighbors(self):
        if not self._min_neighbors:
            self.characteristics()
        return self._min_neighbors


    @property
    def nonzero(self):
        if not self._nonzero:
            self.characteristics()
        return self._nonzero

    @property
    def sd(self):
        if not self._sd:
            self.characteristics()
        return self._sd

    @property
    def asymmetries(self):
        if not self._asymmetries:
            self._asymmetries=self.asymmetry()
        return self._asymmetries

    @property
    def islands(self):
        if not self._islands:
            self.characteristics()
        return self._islands


    @property
    def histogram(self):
        if not self._histogram:
            self.characteristics()
        return self._histogram



    def characteristics(self):
        s0=s1=s2=0.0
        n=len(self.weights)
        col_sum={}
        row_sum={}
        cardinalities={}
        nonzero=0
        for i in self._id_order:
            neighbors_i=self.neighbors[i]
            cardinalities[i]=len(neighbors_i)
            w_i=self.weights[i]
            for j in neighbors_i:
                wij=wji=0
                w_j=self.weights[j]
                neighbors_j=self.neighbors[j]
                if i in neighbors_j:
                    ji=neighbors_j.index(i)
                    wji=w_j[ji]
                if j in neighbors_i:
                    ij=neighbors_i.index(j)
                    wij=w_i[ij]
                v=wij+wji
                if i not in col_sum:
                    col_sum[i]=0
                    row_sum[i]=0
                col_sum[i]+=wji
                row_sum[i]+=wij
                s1+=v*v
                s0+=wij
                nonzero+=1
        s1/=2.0
        s2=sum([(col_sum[i]+row_sum[i])**2 for i in col_sum.keys()])
        self._s2=s2
        self._s1=s1
        self._s0=s0
        self._cardinalities=cardinalities
        cardinalities = cardinalities.values()
        self._max_neighbors=max(cardinalities)
        self._mean_neighbors=sum(cardinalities)/(n*1.)
        self._min_neighbors=min(cardinalities)
        self._sd=np.std(cardinalities)
        self._pct_nonzero=nonzero/(1.0*n*n)
        self._nonzero=nonzero
        self._asymmetries=self.asymmetry()
        self._islands = [i for i,c in self._cardinalities.items() if c==0]
        # connectivity histogram
        ct,bin=np.histogram(cardinalities,range(self._min_neighbors,self._max_neighbors+2))
        self._histogram=zip(bin,ct)


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
        the lexicgraphic order of the keys in the w.weights dictionary. If a specific order
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
        """


        if set(self._id_order) == set(ordered_ids):
            self._id_order=ordered_ids
            self._idx=0
            self._id_order_set=True
            self.neighbor_0_ids={}
            self.__neighbors_0 = False
            self._id2i=False
            self._sparse_set=False
            self._sparse=None
            #self._zero_offset()
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
        if self.__neighbors_0:
            return self.__neighbors_0
        else:
            self.__neighbors_0={}
            id2i=self.id2i
            for id, neigh_list in self.neighbors.iteritems():
                self.__neighbors_0[id]=[id2i[neigh] for neigh in neigh_list] 
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
                self.transformations[value]=weights
                self.weights=weights
                self._reset()
            elif value == "B":
                # binary transformation
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[1.0 for wij in wijs]
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
                      list is column indices of missing elements

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
