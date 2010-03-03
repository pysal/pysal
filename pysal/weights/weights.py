"""
Spatial Weights

"""

__author__  = "Sergio J. Rey <srey@asu.edu> "

from pysal.common import *

# constant for precision
DELTA = 0.0000001

class W(object):
    """Spatial weights

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
                      An ordered list of ids, 
                        defines the order of observations when iterating over W
                        if not set, lexigraphical ordering is used to iterate and
                        the id_order_set property will return False.
                      This can be set after creation by setting the 'id_order' property.

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
                        like neighbors but with zero offset ids, used for
                        alignment in calculating spatial lag
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
    @classmethod
    def fromBinary(cls,data):
        """ creates a new instance of W based on a Dictionary of Sets data structure...
            d = {'a':set(['b','c','d']),'b':set([...]),...}
            returns a new instance of the class, can be called directly
        """
        neighbors={}
        weights={}
        for key in data:
            weights[key] = [1.] * len(data[key])
            neighbors[key] = list(data[key])
            neighbors[key].sort()
        return cls(neighbors,weights)

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
        self.islands=[]
        if id_order == None:
            self._id_order=self.neighbors.keys()
            self._id_order.sort()
            self._id_order_set=False
        else:
            self._id_order=id_order
            self._id_order_set=True
        self.__neighbors_0 = False
        self._idx=0
        self.n=len(self.neighbors)
        self.n_1=self.n-1
        self._characteristics()
        self._transform=None

    def __getitem__(self,key):
        """
        Allow a dictionary like interaction with the weights class.

        Example
        >>> import ContiguityWeights
        >>> w = ContiguityWeights.rook('../examples/10740.shp')
        >>> w[0]
        {1: 1.0, 101: 1.0, 4: 1.0, 5: 1.0, 85: 1.0}
        >>> w = lat2gal()
        >>> w[1]
        {0: 1.0, 2: 1.0, 6: 1.0}
        >>> w[0]
        {1: 1.0, 5: 1.0}
        """
        return dict(zip(self.neighbors[key],self.weights[key]))


    def __iter__(self):
        """Support iteration over weights

        Example:
        >>> w=lat2gal(3,3)
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
        return self

    def next(self):
        if self._idx >= len(self._id_order):
            self._idx=0
            raise StopIteration
        value = self.__getitem__(self._id_order[self._idx])
        self._idx+=1
        return value


    def __set_id_order(self, ordered_ids):
        """Set the iteration order in w.

        W can be iterated over. On construction the iteration order is set to
        the lexicgraphic order of the keys in the w.weights dictionary. If a specific order
        is required it can be set with this method.

        Parameters
        ==========

        ordered_ids : sequence of ids

        Notes
        =====

        ordered_ids is checked against the ids implied by the keys in
        w.weights. If they are not equivalent sets an exception is raised and
        the iteration order is not changed.

        Example:

            >>> w=lat2gal(3,3)
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
        >>> w=lat2gal()
        >>> w.id_order_set
        True
        """
        return self._id_order_set


    @property
    def neighbor_offsets(self):
        """
        Given the current id_order, 
            neighbor_offsets[id] is the offsets of the id's neighrbors in id_order

        Example:
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
            for id in self.neighbors:
                self.__neighbors_0[id]=[self._id_order.index(neigh) for neigh in self.neighbors[id]]
            return self.__neighbors_0


    def get_transform(self):
        """
            Example:
                >>> w=lat2gal()
                >>> w.weights[0]
                [1.0, 1.0]
                >>> w.transform
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
        """Transformations of weights.
        
            Supported transformations include:
                B: Binary 
                R: Row-standardization (global sum=n)
                D: Double-standardization (global sum=1)
                V: Variance stabilizing
                O: Restore original transformation (from instantiation)

            Example:
                >>> w=lat2gal()
                >>> w.weights[0]
                [1.0, 1.0]
                >>> w.transform
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
            self._characteristics()
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
                self._characteristics()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self._characteristics()
                s0=self.s0
                ws=1.0/s0
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[wij*ws for wij in wijs]
                self.transformations[value]=weights
                self.weights=weights
                self._characteristics()
            elif value == "B":
                # binary transformation
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[1.0 for wij in wijs]
                self.transformations[value]=weights
                self.weights=weights
                self._characteristics()
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
                self._characteristics()
            elif value =="O":
                # put weights back to original transformation
                weights={}
                original=self.transformations[value]
                self.weights=original
            else:
                print 'unsupported weights transformation'

    transform = property(get_transform, set_transform)
    

    def _characteristics(self):
        """Calculates properties of W needed for various autocorrelation tests and some
        summary characteristics.
        
        >>> import ContiguityWeights
        >>> w = ContiguityWeights.rook('../examples/10740.shp')
        >>> w[1]
        {0: 1.0, 2: 1.0, 83: 1.0, 4: 1.0}
        >>> w.islands
        [163]
        >>> w[163]
        {}
        >>> w.nonzero
        1002
        >>> w.n
        195
        >>> w.s0
        1002.0
        >>> w.s1
        2004.0
        >>> w.s2
        23528.0
        >>> w.sd
        1.9391533157164347
        >>> w.histogram
        [(0, 1), (1, 1), (2, 4), (3, 20), (4, 57), (5, 44), (6, 36), (7, 15), (8, 7), (9, 1), (10, 6), (11, 0), (12, 2), (13, 0), (14, 0), (15, 1)]
        """
       
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
        self.s2=s2
        self.s1=s1
        self.s0=s0
        self.cardinalities=cardinalities
        cardinalities = cardinalities.values()
        self.max_neighbors=max(cardinalities)
        self.min_neighbors=min(cardinalities)
        self.sd=np.std(cardinalities)
        self.mean_neighbors=sum(cardinalities)/(n*1.)
        self.n=n
        self.pct_nonzero=nonzero/(1.0*n*n)
        self.nonzero=nonzero
        if self.asymmetry():
            self.asymmetric=1
        else:
            self.asymmetric=0
        islands = [i for i,c in self.cardinalities.items() if c==0]
        self.islands=islands
        # connectivity histogram
        ct,bin=np.histogram(cardinalities,range(self.min_neighbors,self.max_neighbors+2))
        self.histogram=zip(bin,ct)

    def asymmetry(self,nonzero=True):
        """Checks for w_{i,j} == w_{j,i} forall w_{i,j}!=0

        Arguments:
            nonzero: (binary) flag to check only that the elements are both
            nonzero. If False, strict equality check is carried out

        Returns:
            asymmetries: a list of 2-tuples with (i,j),(j,i) pairs that are
            asymmetric. If 2-tuple is missing an element then the asymmetry is
            due to a missing weight rather than strict inequality.

        Example Usage:
            >>> neighbors={0:[1,2,3], 1:[1,2,3], 2:[0,1], 3:[0,1]}
            >>> weights={0:[1,1,1], 1:[1,1,1], 2:[1,1], 3:[1,1]}
            >>> w=W(neighbors,weights)
            >>> w.asymmetry()
            [((0, 1), ())]
            >>> weights[1].append(1)
            >>> neighbors[1].insert(0,0)
            >>> w.asymmetry()
            []
            >>> w.transform='r'
            >>> w.asymmetry(nonzero=False)
            [((0, 1), (1, 0)), ((0, 2), (2, 0)), ((0, 3), (3, 0)), ((1, 0), (0, 1)), ((1, 2), (2, 1)), ((1, 3), (3, 1)), ((2, 0), (0, 2)), ((2, 1), (1, 2)), ((3, 0), (0, 3)), ((3, 1), (1, 3))]
            >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
            >>> weights={'first':[1],'second':[1,1],'third':[1]}
            >>> w=W(neighbors,weights)
            >>> w.weights['third'].append(1)
            >>> w.neighbors['third'].append('fourth')
            >>> w.asymmetry()
            [(('third', 'fourth'), ())]

        """


        asymmetries=[]
        for i,neighbors in self.neighbors.iteritems():
            for pos,j in enumerate(neighbors):
                wij=self.weights[i][pos]
                try:
                    wji=self.weights[j][self.neighbors[j].index(i)]
                    if not nonzero and wij!=wji:
                        asymmetries.append(((i,j),(j,i)))
                except:
                    asymmetries.append(((i,j),()))

        return asymmetries


    def full(self):
        """generate a full numpy array

        returns a tuple with first element being the full numpy array and
        second element keys being the ids associated with each row in the
        array.


        Example:
            >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
            >>> weights={'first':[1],'second':[1,1],'third':[1]}
            >>> w=W(neighbors,weights)
            >>> wf,ids=w.full()
            >>> wf
            array([[ 0.,  1.,  1.],
                   [ 1.,  0.,  0.],
                   [ 1.,  0.,  0.]])
            >>> ids
            ['second', 'third', 'first']
        """
        w=np.zeros([self.n,self.n],dtype=float)
        keys=self.neighbors.keys()
        for i,key in enumerate(keys):
            n_i=self.neighbors[key]
            w_i=self.weights[key]
            for j,wij in zip(n_i,w_i):
                c=keys.index(j)
                w[i,c]=wij
        return (w,keys)


    def shimbel(self):
        """find the shmibel matrix for the first order contiguity matrix.
        
            for each observation we store the shortest order between it and
            each of the the other observations.
        >>> w5=lat2gal()
        >>> w5_shimbel=w5.shimbel()
        >>> w5_shimbel[0][24]
        8
        >>> w5_shimbel[0][0:4]
        [-1, 1, 2, 3]
        """

        info={}
        ids=self.neighbors.keys()
        for id in ids:
            s=[0]*self.n
            s[ids.index(id)]=-1
            for j in self.neighbors[id]:
                s[ids.index(j)]=1
            k=1
            flag=s.count(0)
            while flag:
                p=-1
                knext=k+1
                for j in range(s.count(k)):
                    neighbor=s.index(k,p+1)
                    p=neighbor
                    next_neighbors=self.neighbors[p]
                    for neighbor in next_neighbors:
                        nid=ids.index(neighbor)
                        if s[nid]==0:
                            s[nid]=knext
                k=knext
                flag=s.count(0)
            info[id]=s
        return info



    def order(self,kmax=3):
        """Determine the non-redundant order of contiguity up to a specific
        order.

        Implements the algorithm in Anselin and Smirnov (1996)

        currently returns a dictionary of lists with each entry having the
        observation id as the key and the value is a list of order of
        contiguity for the observations in the list (ordered 0 to n-1). a
        negative 1 appears in the ith position

        Example:
            >>> import ContiguityWeights
            >>> w=ContiguityWeights.rook('../examples/10740.shp')
            >>> w3=w.order()
            >>> w3[1][0:5]
            [1, -1, 1, 2, 1]
            >>> 
        """
        ids=self.neighbors.keys()
        info={}
        for id in ids:
            s=[0]*self.n
            s[ids.index(id)]=-1
            for j in self.neighbors[id]:
                s[ids.index(j)]=1
            k=1
            while k < kmax:
                knext=k+1
                if s.count(k):
                    # get neighbors of order k
                    js=[ids[j] for j,val in enumerate(s) if val==k]
                    # get first order neighbors for order k neighbors
                    for j in js:
                        next_neighbors=self.neighbors[j]
                        for neighbor in next_neighbors:
                            nid=ids.index(neighbor)
                            if s[nid]==0:
                                s[nid]=knext
                k=knext
            info[id]=s
        return info

    def higher_order(self,k=3):
        """Contiguity weights object of order k 

        
        
        Implements the algorithm in Anselin and Smirnov (1996)

        Examples
        --------
        >>> w5=lat2gal()
        >>> w5_shimbel=w5.shimbel()
        >>> w5_shimbel[0][24]
        8
        >>> w5_shimbel[0][0:4]
        [-1, 1, 2, 3]
        >>> w5_8th_order=w5.higher_order(8)
        >>> w5_8th_order.neighbors[0]
        [24]
        >>> import ContiguityWeights
        >>> w=ContiguityWeights.rook('../examples/10740.shp')
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

        """

        info=self.order(k)
        ids=info.keys()
        neighbors={}
        weights={}
        for id in ids:
            nids=[ids[j] for j,order in enumerate(info[id]) if order==k]
            neighbors[id]=nids
            weights[id]=[1.0]*len(nids)
        return W(neighbors,weights)


def lat2gal(nrows=5,ncols=5,rook=True,id_type='int'):
    """Create a GAL structure for a regular lattice.

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

    >>> w9=lat2gal(3,3)
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
    """Construct spatial weights for regime neighbors.

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
    """Combinations of size n taken from items

    Arguments
    ---------

    items : sequence
    n     : integer
            size of combinations to take from items

    Returns
    -------
    generator of combinations of size n taken from items

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

    
if __name__ == "__main__":

    import doctest
    doctest.testmod()
