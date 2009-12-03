"""
Spatial weights for PySAL

"""

__author__  = "Sergio J. Rey <srey@asu.edu> "

import unittest
import numpy as num
import numpy.linalg as la
import time
import math
import pysal
import copy

# constant for precision
DELTA = 0.0000001

class W(object):
    """Spatial weights"""
    @classmethod
    def fromBinary(cls,data):
        """ creates a new instance of W based on a Dictionary of Sets data structure...
            d = {'a':set(['b','c','d']),'b':set([...]),...}
            returns a new instance of the class, can be called directly
        """
        d = {'neighbors':{},'weights':{}}
        for key in data:
            d['weights'][key] = [1] * len(data[key])
            d['neighbors'][key] = list(data[key])
            d['neighbors'][key].sort()
        return cls(d)



    def __init__(self,data):
        """Construct a spatial weights object

        Parameters
        ==========

        data: dictionary with two entries
            neighbors: list of neighbors
            weights: list of weights

            such that:

            neighbors[1]=[2,4]
            weights[1]=[w_{1,2}, w_{1,4}]
            .
            .
            neighbors[4]=[1,7,10]
            weights[4]=[w_{4,1}, w_{4,7}, w_{4,10}]

            and w_{i,j} are the weights which can be general or
            binary.


        Attributes:
        ===========

        asymmetric: Flag for any asymmetries (see
        method asymmetry for details), false if none.

        cardinalities: dictionary of cardinalities 

        islands: list of ids that have no neighbors

        max_neighbors: maximum cardinality (int)

        min_neighbors: minimum cardinality (int)

        mean_neighbors: average cardinality (float)

        n: number of observations (int)

        neighbors: dictionary of neighbor ids. key is observation id
        and value is list of neighboring observation ids, with position
        corresponding to the same position in weights (see weights)

        nonzero: number of nonzero weights

        pct_nonzero: percentage of all weights that are nonzero

        s0: sum of all weights 

        s1: trace of ww

        s2: trace of w'w

        sd: standard deviation of number of neighbors (float)

        transform: property for weights transformation. can be used to
        get and set weights transformation. 

        transformations: dictionary of transformed weights. key is
        transformation type, value are weights

        weights: dictionary of currently specified transformed
        weights. key is observation id, value is list of transformed
        weights in order of neighbor ids (see neighbors).

        Private attributes
        ------------------

        _idx: index for iterator
        _id_order: order of ids for iterator
        _transform: weights transformation

                

        Methods:
        ========

        asymmetry: checks if there are any asymmetries in the weights.
        The default is to check for non-zero symmetries only, but
        stricter value symmetries can also be checked.

        characteristics: calculates summary properties of the weights

        full: returns a full nxn numpy array

        higher_order: returns the higher order (k) contiguity matrix

        lag: calculate spatial lag of an array

        order: determines the non-redundant order of contiguity for
        i,j up to a given level of contiguity

        set_transform: transform the weights.  Options include "B": binary,
        "W": row-standardized (row sum to 1), "D": doubly-standardized
        (global sum to 1), "V": variance stabilizing, 
        "O": original weights

        shimbel: finds the shimbel matrix for the first order
        contiguity matrix


        Example:
        ========

                >>> neighbors={0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
                >>> weights={0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
                >>> data={'neighbors':neighbors,'weights':weights}
                >>> w=W(data)
                >>> w.pct_nonzero
                0.29629629629629628
        """

        if 'weights' not in data:
            weights=dict([(i,[1]*len(wi)) for i,wi in data.items()])
            neighbors=data
            data={'neighbors':neighbors,'weights':weights}

        self.transformations={}
        self.weights=data['weights']
        self.neighbors=data['neighbors']
        self.transformations['O']=self.weights #original weights
        self.islands=[]
        self.characteristics()
        self._transform=None
        self._id_order=self.neighbors.keys()
        self._idx=0
        self.n=len(self.neighbors)
        self.n_1=self.n-1
        
    def __getitem__(self,key):
        """
        Allow a dictionary like interaction with the weights class.

        Example
        >>> import ContiguityWeights
        >>> w = ContiguityWeights.rook('../examples/10740.shp')
        >>> w[1]
        {2: 1, 102: 1, 86: 1, 5: 1, 6: 1}
        >>> w = lat2gal()
        >>> w[1]
        {0: 1, 2: 1, 6: 1}
        >>> w[0]
        {1: 1, 5: 1}
        """
        return dict(zip(self.neighbors[key],self.weights[key]))


    def __iter__(self):
        """Support iteration over weights


        Example:
        >>> w=lat2gal(3,3)
        >>> for wi in w:
        ...     print wi
        ...     
        {1: 1, 3: 1}
        {0: 1, 2: 1, 4: 1}
        {1: 1, 5: 1}
        {0: 1, 4: 1, 6: 1}
        {1: 1, 3: 1, 5: 1, 7: 1}
        {8: 1, 2: 1, 4: 1}
        {3: 1, 7: 1}
        {8: 1, 4: 1, 6: 1}
        {5: 1, 7: 1}
        >>> w=lat2gal(3,3)
        >>> for i,wi in enumerate(w):
        ...     print i,wi
        ...     
        0 {1: 1, 3: 1}
        1 {0: 1, 2: 1, 4: 1}
        2 {1: 1, 5: 1}
        3 {0: 1, 4: 1, 6: 1}
        4 {1: 1, 3: 1, 5: 1, 7: 1}
        5 {8: 1, 2: 1, 4: 1}
        6 {3: 1, 7: 1}
        7 {8: 1, 4: 1, 6: 1}
        8 {5: 1, 7: 1}
        >>> 
        """
        return self

    def next(self):
        if self._idx >= len(self._id_order):
            raise StopIteration
        value = self.__getitem__(self._id_order[self._idx])
        self._idx+=1
        return value


    def set_id_order(self, ordered_ids):
        """reorder the ids in w


        Example:
            >>> neighbors={'a': ['b'],'b':['a','c'],'c':['b','d'], 'd':['c']}
            >>> w=W(neighbors)
            >>> y=num.array([0,10,100,1000])
            >>> w.id_order=['a','b','c','d']
            >>> yl=w.lag(y)
            >>> y
            array([   0,   10,  100, 1000])
            >>> yl
            array([   10.,   100.,  1010.,   100.])
            >>> y1=y[[1,0,3,2]]
            >>> y1
            array([  10,    0, 1000,  100])
            >>> w.id_order=['b','a','d','c']
            >>> y1l=w.lag(y1)
            >>> y1l
            array([  100.,    10.,   100.,  1010.])
            >>> 
        """


        if set(self._id_order) == set(ordered_ids):
            self._id_order=ordered_ids
            self.refresh()
        else:
            print 'assignment not aligned with W ids'

    def get_id_order(self):
        """returns the ids for the observations in the order in which they
        would be encountered if iterating over the weights."""
        return self._id_order

    id_order=property(get_id_order, set_id_order)

    def refresh(self):
        """move iterator back to beginning


        Example:

            >>> from weights import *
            >>> w=lat2gal(2,2)
            >>> for wi in w:
            ...     print wi
            ...     
            {1: 1, 2: 1}
            {0: 1, 3: 1}
            {0: 1, 3: 1}
            {1: 1, 2: 1}
            >>> for wi in w:
            ...     print wi
            ...     
            >>> w.refresh()
            >>> for wi in w:
            ...     print wi
            ...     
            {1: 1, 2: 1}
            {0: 1, 3: 1}
            {0: 1, 3: 1}
            {1: 1, 2: 1}
            >>> 
            


        """
        self._idx=0

    def lag(self,y):
        """Calculates the spatial lag for the variable y


        Argument:

            y -  numpy array (n,)

        Returns:

            yl -  numpy array (n,) for the spatial lag.


        Assumes id_order in weights is aligned with order in y. User can
        reorder W (as in example below) or reorder y

        Example:

        >>> w=lat2gal(3,3)
        >>> y=num.arange(9)
        >>> yl=w.lag(y)
        >>> yl
        array([  4.,   6.,   6.,  10.,  16.,  14.,  10.,  18.,  12.])
        >>> w.transform='w'
        """
        if self.n !=len(y):
            print 'w.lag(y):  w and y not conformable'
            return 0
        else:
            self.refresh()
            yl=num.zeros(y.shape,'float')
            for i,wi in enumerate(self):
                for j,wij in wi.items():
                    yl[i]+=wij*y[self.id_order.index(j)]
            return yl


    def get_transform(self):
        """
            Example:
                >>> w=lat2gal()
                >>> w.weights[0]
                [1, 1]
                >>> w.transform
                >>> w.transform='w'
                >>> w.weights[0]
                [0.5, 0.5]
                >>> w.transform='b'
                >>> w.weights[0]
                [1, 1]
                >>> 
        """
        return self._transform

    def set_transform(self, value="B"):
        """Transformations of weights.
        
            Supported transformations include:
                B: Binary 
                W: Row-standardization (global sum=n)
                D: Double-standardization (global sum=1)
                V: Variance stabilizing
                O: Restore original transformation (from instantiation)

            Example:
                >>> w=lat2gal()
                >>> w.weights[0]
                [1, 1]
                >>> w.transform
                >>> w.transform='w'
                >>> w.weights[0]
                [0.5, 0.5]
                >>> w.transform='b'
                >>> w.weights[0]
                [1, 1]
                >>> 
        """
        value=value.upper()
        self._transform = value
        if self.transformations.has_key(value):
            self.weights=self.transformations[value]
            self.characteristics()
        else:
            if value == "W": 
                # row standardized weights
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    row_sum=sum(wijs)*1.0
                    weights[i]=[wij/row_sum for wij in wijs]
                self.transformations[value]=weights
                self.weights=weights
                self.characteristics()
            elif value == "D":
                # doubly-standardized weights
                # update current chars before doing global sum
                self.characteristics()
                s0=self.s0
                ws=1.0/s0
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[wij*ws for wij in wijs]
                self.transformations[value]=weights
                self.weights=weights
                self.characteristics()
            elif value == "B":
                # binary transformation
                weights={}
                for i in self.weights:
                    wijs = self.weights[i]
                    weights[i]=[1 for wij in wijs]
                self.transformations[value]=weights
                self.weights=weights
                self.characteristics()
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
                self.characteristics()
            elif value =="O":
                # put weights back to original transformation
                weights={}
                original=self.transformations[value]
                self.weights=original
            else:
                print 'unsupported weights transformation'

    transform = property(get_transform, set_transform)
    

    def characteristics(self):
        """properties of W needed for various autocorrelation tests and some
        summary characteristics.
        
        >>> import ContiguityWeights
        >>> w = ContiguityWeights.rook('../examples/10740.shp')
        >>> w[1]
        {2: 1, 102: 1, 86: 1, 5: 1, 6: 1}
        >>> w.islands
        [164]
        >>> w[164]
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
        23528
        >>> w.sd
        1.9391533157164347
        """
       
        s0=s1=s2=0.0
        n=len(self.weights)
        col_sum={}
        row_sum={}
        cardinalities={}
        nonzero=0
        for i in self.weights.keys():
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
        self.sd=num.std(cardinalities)
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
            >>> data={'neighbors':neighbors,'weights':weights}
            >>> w=W(data)
            >>> w.asymmetry()
            [((0, 1), ())]
            >>> weights[1].append(1)
            >>> neighbors[1].insert(0,0)
            >>> w.asymmetry()
            []
            >>> w.transform='w'
            >>> w.asymmetry(nonzero=False)
            [((0, 1), (1, 0)), ((0, 2), (2, 0)), ((0, 3), (3, 0)), ((1, 0), (0, 1)), ((1, 2), (2, 1)), ((1, 3), (3, 1)), ((2, 0), (0, 2)), ((2, 1), (1, 2)), ((3, 0), (0, 3)), ((3, 1), (1, 3))]
            >>> neighbors={'first':['second'],'second':['first','third'],'third':['second']}
            >>> weights={'first':[1],'second':[1,1],'third':[1]}
            >>> data={'neighbors':neighbors,'weights':weights}
            >>> w=W(data)
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
            >>> data={'neighbors':neighbors,'weights':weights}
            >>> w=W(data)
            >>> wf,ids=w.full()
            >>> wf
            array([[ 0.,  1.,  1.],
                   [ 1.,  0.,  0.],
                   [ 1.,  0.,  0.]])
            >>> ids
            ['second', 'third', 'first']
        """
        w=num.zeros([self.n,self.n],dtype=float)
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
            [-1, 1, 2, 2, 1]
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
        """Return contiguity weights object of order k (default k=3)
        
        
        Implements the algorithm in Anselin and Smirnov (1996)

        Example:
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
            {2: 1, 102: 1, 86: 1, 5: 1, 6: 1}
            >>> w2[1]
            {147: 1, 3: 1, 4: 1, 101: 1, 7: 1, 40: 1, 41: 1, 10: 1, 103: 1, 104: 1, 19: 1, 84: 1, 85: 1, 100: 1, 91: 1, 92: 1, 94: 1}
            >>> w[147]
            {163: 1, 100: 1, 165: 1, 102: 1, 140: 1, 145: 1, 146: 1, 148: 1, 85: 1}
            >>> w[85]
            {96: 1, 102: 1, 140: 1, 147: 1, 86: 1, 94: 1}
            >>> 

        """

        info=self.order(k)
        data={}
        ids=info.keys()
        neighbors={}
        weights={}
        for id in ids:
            nids=[ids[j] for j,order in enumerate(info[id]) if order==k]
            neighbors[id]=nids
            weights[id]=[1]*len(nids)
        data['weights']=weights
        data['neighbors']=neighbors
        return W(data)

def lat2gal(nrows=5,ncols=5,rook=True):
    """Create a GAL structure for a regular lattice.

    Arguments:

        nrows = number of rows
        ncols = number of columns
        rook = boolean for type of matrix. Default is rook. For queen set
        rook=False

    Returns:

        w = instance of spatial weights class W

    Notes:

        Observations are row ordered: first k observations are in row 0, next
        k in row 1, and so on.

    Example:

        >>> w9=lat2gal(3,3)
        >>> w9.pct_nonzero
        0.29629629629629628
        >>> w9[0]
        {1: 1, 3: 1}
        >>> w9[3]
        {0: 1, 4: 1, 6: 1}
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

    d={}
    neighbors={}
    weights={}
    for key in w:
        weights[key]=[1]*len(w[key])
    d['neighbors']=w
    d['weights']=weights
    return W(d)

def regime_weights(regimes):
    """Construct spatial weights for regime neighbors.

    Block contiguity structures are relevant when defining neighbor relations
    based on membership in a regime. For example, all counties belonging to
    the same state could be defined as neighbors.

    Arguments
    =========
    regimes - list or array
           ids of which regime an observation belongs to

    Returns
    =======

    W - spatial weights instance

    Example
    =======
    >>> regimes=num.ones(25)
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
    >>> y=num.arange(25)
    >>> w.lag(y)
    array([  65.,   64.,   63.,   62.,   61.,   60.,   59.,   58.,   57.,
             56.,  135.,  134.,  133.,  132.,  131.,  130.,  129.,  128.,
            127.,  126.,   45.,   69.,   68.,   67.,   66.])
    >>> w.transform='w'
    >>> w.lag(y)
    array([  6.5       ,   6.4       ,   6.3       ,   6.2       ,
             6.1       ,   6.        ,   5.9       ,   5.8       ,
             5.7       ,   5.6       ,  15.        ,  14.88888889,
            14.77777778,  14.66666667,  14.55555556,  14.44444444,
            14.33333333,  14.22222222,  14.11111111,  14.        ,
             4.5       ,  23.        ,  22.66666667,  22.33333333,  22.        ])
    >>> regimes=['n','n','s','s','e','e','w','w','e']
    >>> n=len(regimes)
    >>> w=regime_weights(regimes)
    >>> w.neighbors
    {0: [1], 1: [0], 2: [3], 3: [2], 4: [5, 8], 5: [4, 8], 6: [7], 7: [6], 8: [4, 5]}
    >>> y=num.arange(n)
    >>> w.lag(y)
    array([  1.,   0.,   3.,   2.,  13.,  12.,   7.,   6.,   9.])
    """ 
    region_ids=list(set(regimes))
    regimes=num.array(regimes)
    regions=[num.nonzero(regimes==region)[0] for region in region_ids]
    neighbors={}
    weights={}
    for region in regions:
        for i in region:
            neighbors[i]=[j for j in region if j!=i]
            weights[i]=[1.]*len(neighbors[i])

    return W({'neighbors':neighbors,'weights':weights})

class __TestWeights(unittest.TestCase):

    def setUp(self):
        pass

    def test_lat2gal(self):
        w=lat2gal(10,10)
        self.assertEquals(w.s0,360.0)
        self.assertEquals(w.s1,720.0)
        self.assertEquals(w.s2,5312.0)

    def test_full(self):
        w=lat2gal(50,50)
        wf=w.full()[0]
        self.assertEquals(w.s0,9800.0)
        self.assertEquals(sum(sum(wf)),9800.0)
        self.assertEquals(w.neighbors[401],[351, 400, 451, 402])
        j=[351, 400, 451, 402]
        self.assertEquals(wf[401,j].tolist(),[1.0, 1.0, 1.0, 1.0])
        j=[i+1 for i in j]
        self.assertEquals(wf[401,j].tolist(),[0.0, 0.0, 0.0, 0.0])

    def test_higher_order(self):
        w5=lat2gal()
        w5_shimbel=w5.shimbel()
        self.assertEquals(w5_shimbel[0][24],8)
        self.assertEquals(w5_shimbel[0][0:4],[-1, 1, 2, 3])
        w5_8th_order=w5.higher_order(8)
        self.assertEquals( w5_8th_order.neighbors[0],[24])
    
if __name__ == "__main__":

    import unittest
    import doctest
    import weights

    # use unittest to test doctests
    suite = unittest.TestSuite()
    for mod in [weights]:
            suite.addTest(doctest.DocTestSuite(mod))
            runner = unittest.TextTestRunner()
            runner.run(suite)
    # regular unittest
    unittest.main()



