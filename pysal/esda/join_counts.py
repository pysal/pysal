"""
Spatial autocorrelation for binary attributes

"""
__author__  = "Sergio J. Rey <srey@asu.edu> "

import pysal
import numpy as np

__all__ = ['Join_Counts']

class Join_Counts:
    """Binary Join Counts
    

    Parameters
    ----------

    y               : array
                      binary variable measured across n spatial units
    w               : W
                      spatial weights instance


    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
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



    Examples
    --------

    Replicate example from anselin and rey

    >>> import numpy as np
    >>> w=pysal.lat2W(4,4)
    >>> y=np.ones(16)
    >>> y[0:8]=0
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
    >>> 
    """
    def __init__(self,y,w):
        w.transformation='b' # ensure we have binary weights
        self.w=w
        self.y=y
        b=sum(y)
        bb=pysal.lag_spatial(w,y)
        self.B=b
        self.W=w.n-b
        self.J=w.s0/2.
        self.bb=sum(y*bb)/2.
        yw=1-y;
        ww=pysal.lag_spatial(w,yw)
        self.ww=sum(yw*ww)/2.
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
        
def _test():
    import doctest
    doctest.testmod(verbose=True)



if __name__ == '__main__':
    _test()

