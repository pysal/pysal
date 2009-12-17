"""
Distance Based Spatial Weights for PySAL



"""

__author__  = "Sergio J. Rey <srey@asu.edu> "


from pysal.weights import W
from pysal.common import *

class InverseDistance(W):
    """Creates spatial weights based on inverse distance """
    def __init__(self,data,p=2,row_standardize=True):
        """

        Arguments:
            data: n by m array of attribute data, n observations on m
            attributes


            p: Minkowski p-norm distance metric parameter:
                1<=p<=infinity
                2: Euclidean distance
                1: Manhattan distance


            row_standardize: (binary) True if weights are to be row standardized
            (default) False if not

        Example Usage:
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
    """Creates contiguity matrix based on k nearest neighbors"""
    def __init__(self,data,k=2,p=2):
        """

        Arguments:
            data: n by m array of attribute data, n observations on m
            attributes


            k: number of nearest neighbors

            p: Minkowski p-norm distance metric parameter:
                1<=p<=infinity
                2: Euclidean distance
                1: Manhatten distance


        Example Usage:
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
            >>> points=[(10,10),(20,10),(40,10),(15,20),(30,20),(30,30)]
        """
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
    """Creates contiguity matrix based on distance band"""
    def __init__(self,data,threshold,p=2):
        """
        Arguments:

            data: n by m array of attribute data, n observations on m
            attributes


            threshold: distance band

            p: Minkowski p-norm distance metric parameter:
                1<=p<=infinity
                2: Euclidean distance
                1: Manhatten distance

        Example Usage:
            >>> points=[(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]
            >>> w=DistanceBand(points,threshold=11.2)
            >>> w.weights
            {0: [1, 1], 1: [1, 1], 2: [], 3: [1, 1], 4: [1], 5: [1]}
            >>> w.neighbors
            {0: [1, 3], 1: [0, 3], 2: [], 3: [0, 1], 4: [5], 5: [4]}
            >>> 
        """
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




        

if __name__ == '__main__':
    import doctest
    doctest.testmod()

