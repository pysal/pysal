"""
Spatial autocorrelation for binary attributes

"""
__author__  = "Sergio J. Rey <srey@asu.edu> , Luc Anselin <luc.anselin@asu.edu>"

import pysal
import numpy as np

__all__ = ['Join_Counts']

PERMUTATIONS = 999

class Join_Counts:
    """Binary Join Counts
    

    Parameters
    ----------

    y               : array
                      binary variable measured across n spatial units
    w               : W
                      spatial weights instance
    permutations    : int
                      number of random permutations for calculation of pseudo-p_values

    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of permutations
    bb           : float
                   number of black-black joins
    ww           : float
                   number of white-white joins
    bw           : float
                   number of black-white joins
    J            : float
                   number of joins
    Ebb          : float
                   expected value of bb under free sampling
    Eww          : float
                   expected value of ww under free sampling
    Ebw          : float
                   expected value of bw under free sampling
    Vbb          : float
                   variance of bb under free sampling
    Vww          : float
                   variance of ww under free sampling
    Vbw          : float
                   variance of bw under free sampling
    zbb          : float
                   z-value for bb under free sampling
    zww          : float
                   z-value for ww under free sampling
    zbw          : float
                   z-value for bw under free sampling
    sim_bb       : array (if permutations>0)
                   vector of bb values for permutated samples
    p_sim_bb     : array (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed bb is extreme
                                it is either extremely greater or extremely lower


    Examples
    --------

    Replicate example from anselin and rey

    >>> import numpy as np
    >>> w=pysal.lat2W(4,4)
    >>> y=np.ones(16)
    >>> y[0:8]=0
    >>> np.random.seed(10)
    >>> jc=Join_Counts(y,w)
    >>> jc.bb
    10.0
    >>> jc.zbb
    1.2060453783110545
    >>> jc.bw
    4.0
    >>> jc.zbw
    -3.2659863237109046
    >>> jc.Ebw
    12.0
    >>> jc.bw
    4.0
    >>> jc.Vbw
    6.0
    >>> np.sqrt(jc.Vbw)
    2.4494897427831779
    >>> jc.p_sim_bb
    0.0030000000000000001
    >>> np.mean(jc.sim_bb)
    5.6396396396396398
    >>> np.max(jc.sim_bb)
    10.0
    >>> 
    """
    def __init__(self,y,w,permutations = PERMUTATIONS):
        w.transformation='b' # ensure we have binary weights
        self.w=w
        self.y=y
        self.permutations = permutations
        b=sum(y)*1.
        self.B=b
        self.W=w.n-b
        self.J=w.s0/2.
        self.bb = self.__calc(self.y)
        yw=1-y;
        self.ww = self.__calc(yw)
        self.bw=self.J-(self.ww+self.bb)
        pb= (self.B/w.n)
        pw= (self.W/w.n)
        pb2=pb**2
        pw2=pw**2
        Ebb=self.J * pb2
        Eww=self.J * pw2
        Ebw=2. * self.J * pb * pw
        self.Ebb=Ebb
        self.Eww=Eww
        self.Ebw=Ebw
        self.pb=pb
        self.pw=pw
        ks=w.cardinalities.values()
        m=sum([k*(k-1) for k in ks])/2.
        self.m=m
        self.Vbb=self.J * pb2 + 2*m*pb2*pb - (self.J + 2*m)*pb2*pb2
        self.Vww=self.J * pw2 + 2*m*pw2*pw - (self.J + 2*m)*pw2*pw2
        self.Vbw=2*(self.J + m)*pb*pw - 4*(self.J+2*m)*pb2*pw2
        self.zbb=(self.bb-self.Ebb)/np.sqrt(self.Vbb)
        self.zww=(self.ww-self.Eww)/np.sqrt(self.Vww)
        self.zbw=(self.bw-self.Ebw)/np.sqrt(self.Vbw)
        
        if permutations:
            sim = [self.__calc(np.random.permutation(self.y)) \
                 for i in xrange(permutations)]
            self.sim_bb = sim_bb = np.array(sim)
            above = sim_bb >= self.bb
            larger = sum(above)
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim_bb = (larger + 1.)/(permutations + 1.)
        
    def __calc(self,z):
        zl = pysal.lag_spatial(self.w,z)
        jc = sum(z*zl)/2.0
        return jc

        
def _test():
    import doctest
    doctest.testmod(verbose=True)



if __name__ == '__main__':
    _test()

