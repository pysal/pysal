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

from numpy import array
# constants for weight file type
weightFtypes = ['gal','gwt']
weightTypes = ['raw','binary','invdistance','triangle','epanech','bisquare']
# constant for precision
DELTA = 0.0000001

class W(object):
    """Spatial weights"""
    def __init__(self,data):
        """
            Arguments:
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

                asymmetric: Flag for any asymmetries (see
                method asymmetry for details), false if none.

                cardinalities: list of cardinalities (number of neighbors for
                each observation)

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

                

            Methods:

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

                transpose: returns transpose as a W object



        """

        self.transformations={}
        self.weights=data['weights']
        self.neighbors=data['neighbors']
        if 'new_ids' in data:
            self.new_ids=data['new_ids']
        if 'original_ids' in data:
            self.old_ids=data['original_ids']
        self.transformations['O']=self.weights #original weights
        self.islands=[]
        self.characteristics()
        self._transform=None
        
    def get_transform(self):
        return self._transform

    def set_transform(self, value="B"):
        """Transformations of weights.
        
            Supported transformations include:
                B: Binary 
                W: Row-standardization (global sum=n)
                D: Double-standardization (global sum=1)
                V: Variance stabilizing
                O: Restore original transformation (from instantiation)
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
        
        note: does not handle islands (explicitly) (yet)
        """
       
        s0=s1=s2=0.0
        n=len(self.weights)
        col_sum=[0.0]*n
        row_sum=[0.0]*n
        cardinalities=[0]*n
        nonzero=0
        for i in self.weights:
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
                col_sum[i]+=wji
                row_sum[i]+=wij
                s1+=v*v
                s0+=wij
                nonzero+=1
        s1/=2.0
        s2=sum([(col_sum[i]+row_sum[i])**2 for i in xrange(n)])
        self.s2=s2
        self.s1=s1
        self.s0=s0
        self.cardinalities=cardinalities
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


    def lag(self,y):
        "spatial lag"
        yl=num.zeros(y.shape)
        for i in self.weights:
            neighbors=self.neighbors[i]
            wijs=self.weights[i]
            z=zip(neighbors,wijs)
            yl[i]= sum([y[j]*w for j,w in z])
        return yl

    def full(self):
        "returns weights a full nxn numpy array"
        w=num.zeros([self.n,self.n],dtype=float)
        for i in self.neighbors:
            n_i=self.neighbors[i]
            w_i=self.weights[i]
            for j,wij in zip(n_i,w_i):
                w[i,j]=wij
        return w

    def transpose(self):
        """returns transpose as a W object
        """
        
        weights={}
        neighbors={}
        for id in xrange(self.n):
            for j,neighbor in enumerate(self.neighbors[id]):
                if neighbor not in neighbors:
                    neighbors[neighbor]=[]
                    weights[neighbor]=[]
                neighbors[neighbor].append(id)
                weights[neighbor].append(self.weights[id][j])
        data={'neighbors':neighbors, 'weights':weights}
        return W(data)

    def shimbel(self):
        """find the shmibel matrix for the first order contiguity matrix.
        
            for each observation we store the shortest order between it and
            each of the the other observations.
        """

        info={}
        for i in xrange(self.n):
            s=[0]*self.n
            s[i]=-1
            for j in self.neighbors[i]:
                s[j]=1
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
                        if s[neighbor]==0:
                            s[neighbor]=knext
                k=knext
                flag=s.count(0)
            info[i]=s
        return info



    def order(self,kmax=3):
        """Determine the non-redundant order of contiguity up to a specific
        order.

        Implements the algorithm in Anselin and Smirnov (1996)

        currently returns a dictionary of lists with each entry having the
        observation id as the key and the value is a list of order of
        contiguity for the observations in the list (ordered 0 to n-1). a
        negative 1 appears in the ith position

        Note:
            work in progress

        """
        info={}
        for i in xrange(self.n):
            s=[0]*self.n
            s[i]=-1
            for j in self.neighbors[i]:
                s[j]=1
            k=1
            while k < kmax:
                p=-1
                knext=k+1
                for j in range(s.count(k)):
                    neighbor=s.index(k,p+1)
                    p=neighbor
                    next_neighbors=self.neighbors[p]
                    for neighbor in next_neighbors:
                        if s[neighbor]==0:
                            s[neighbor]=knext
                k=knext
            info[i]=s
        return info

    def higher_order(self,k=3):
        """Return contiguity weights object of order k (default k=3)
        
        
        Implements the algorithm in Anselin and Smirnov (1996)
        Example Usage:

        >>> w5=lat2gal()
        >>> w5_shimbel=w5.shimbel()
        >>> w5_shimbel[0][24]
        8
        >>> w5_shimbel[0][0:4]
        [-1, 1, 2, 3]
        >>> w5_8th_order=w5.higher_order(8)
        >>> w5_8th_order.neighbors[0]
        [24]
        >>> 

        """

        info=self.order(k)
        data={}
        ids=info.keys()
        ids.sort()
        n=len(ids)
        rn=range(n)
        neighbors={}
        weights={}
        for i in ids:
            nids=[j for j,order in enumerate(info[i]) if order==k]
            neighbors[i]=nids
            weights[i]=[1]*len(nids)
        data['weights']=weights
        data['neighbors']=neighbors
        return W(data)


class Wd(dict):
    """ A basic Weights data structure """
    @classmethod
    def fromDictSets(cls,d):
        """ populates Wd from a dictionary of Sets
            d = {'a':set(['b','c','d']),'b':set([...]),...}
            returns a new instance of the class, can be called directly
        """
        w = cls()
        for key in d:
            if key not in w:
                w[key] = {}
            for neighbor in d[key]:
                w[key][neighbor] = 1
        return w
    @classmethod
    def fromDictLists(cls,d):
        """ populates Wd from a dictionary of Sets
            d = {'a':['b','c','d'],'b':[...],...}
            returns a new instance of the class, can be called directly
        """
        w = cls()
        for key in d:
            if key not in w:
                w[key] = {}
            for neighbor in d[key]:
                w[key][neighbor] = 1
        return w
        
def lat2gal(nrows=5,ncols=5,rook=True, file=None):
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
        wf=w.full()
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

