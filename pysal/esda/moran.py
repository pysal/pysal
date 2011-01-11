"""
Moran's I Spatial Autocorrelation Statistics

"""
__author__  = "Sergio J. Rey <srey@asu.edu>"
from pysal.common import *
from pysal.weights.spatial_lag import lag_spatial as slag

__all__=["Moran","Moran_Local","Moran_BV","Moran_BV_matrix"]


PERMUTATIONS=999

class Moran:
    """Moran's I Global Autocorrelation Statistic
    
    Parameters
    ----------

    y               : array
                      variable measured across n spatial units
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
    I            : float
                   value of Moran's I
    EI           : float
                   expected value under normality assumption
    VI_norm      : float
                   variance of I under normality assumption
    seI_norm     : float
                   standard deviation of I under normality assumption
    z_norm       : float
                   z-value of I under normality assumption
    p_norm       : float
                   p-value of I under normality assumption (1-tailed)
    VI_rand      : float
                   variance of I under randomization assumption
    seI_rand     : float
                   standard deviation of I under randomization assumption
    z_rand       : float
                   z-value of I under randomization assumption
    p_rand       : float
                   p-value of I under randomization assumption (1-tailed)
    sim          : array (if permutations>0)
                   vector of I values for permutated samples
    p_sim        : array (if permutations>0)
                   p-value based on permutations
    EI_sim       : float (if permutations>0)
                   average value of I from permutations
    VI_sim       : float (if permutations>0)
                   variance of I from permutations
    seI_sim      : float (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float (if permutations>0)
                   p-value based on standard normal approximation from

    Examples
    --------
    >>> import pysal
    >>> w=pysal.open("../examples/stl.gal").read()
    >>> f=pysal.open("../examples/stl_hom.txt")
    >>> y=np.array(f.by_col['HR8893'])
    >>> mi=Moran(y,w)
    >>> "%7.5f"%mi.I
    '0.24366'
    >>> mi.EI
    -0.012987012987012988
    >>> mi.p_norm
    0.00027147862770937614

    SIDS example replicating OpenGeoda

    >>> w=pysal.open("../examples/sids2.gal").read()
    >>> f=pysal.open("../examples/sids2.dbf")
    >>> SIDR=np.array(f.by_col("SIDR74"))
    >>> mi=pysal.Moran(SIDR,w)
    >>> "%6.4f"%mi.I
    '0.2477'
    >>> mi.p_norm
    0.0001158330781489969
    
    """
    def __init__(self,y,w,transformation="r",permutations=PERMUTATIONS):
        self.y=y
        w.transform=transformation
        self.w=w
        self.permutations=permutations
        self.__moments()
        self.I = self.__calc(self.z)
        self.z_norm = (self.I - self.EI)/self.seI_norm
        self.p_norm = 2.0*(1-stats.norm.cdf(np.abs(self.z_norm)))
        self.z_rand = (self.I - self.EI)/self.seI_rand
        self.p_rand = 2.0*(1-stats.norm.cdf(np.abs(self.z_rand)))


        if permutations:
            sim=[self.__calc(np.random.permutation(self.z)) \
                 for i in xrange(permutations)]
            self.sim=sim
            self.p_sim=(sum(sim >= self.I)+1.)/(permutations+1.)
            self.EI_sim = sum(sim)/permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim=(self.I - self.EI_sim)/self.seI_sim
            self.p_z_sim=2.0*(1-stats.norm.cdf(np.abs(self.z_sim)))

    def __moments(self):
        self.n=len(self.y)
        y=self.y
        #z=(y-y.mean())/y.std()
        z=y-y.mean()
        self.z=z
        self.z2ss=sum(z*z)
        self.EI = -1./(self.n-1)
        n=self.n
        s1=self.w.s1
        s0=self.w.s0
        s2=self.w.s2
        s02=s0*s0
        v_num=n*n*s1 - n*s2 + 3* s0*s0
        v_den=(n-1)*(n+1)*s0*s0
        self.VI_norm = v_num/v_den - (1.0/(n-1))**2
        self.seI_norm = self.VI_norm**(1/2.)

        k = (1/(sum(z**4)) * ((sum(z**2))**2))
        vi = (1/(((n-1)**3)*s02)) * ((n*((n*n-3*n+3)*s1 - n*s2+3*s02)) \
                - (k*((n*n-n)*s1-2*n*s2+6*s02)))
        self.VI_rand = vi
        self.seI_rand = vi**(1/2.)


    def __calc(self,z):
        zl=slag(self.w,z)
        self.inum=sum(self.z*zl)
        return self.n/self.w.s0 * sum(self.z*zl)/self.z2ss

class Moran_BV:
    """Bivariate Moran's I
    
 
    
    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        (wy will be on y axis)
    w : W
        weight instance assumed to be aligned with y
    transformation  : string
                      weights transformation, default is row-standardized "r".
                      Other options include "B": binary, "D":
                      doubly-standardized, "U": untransformed (general weights),
                      "V": variance-stabilizing.
    permutations    : int
                      number of random permutations for calculation of pseudo-p_values


    Attributes
    ----------
    zx            : array
                    original x variable standardized by mean and std
    zy            : array
                    original y variable standardized by mean and std
    w             : W 
                    original w object
    permutation   : int
                    number of permutations
    I             : float
                    value of bivariate Moran's I
    sim           : array (if permutations>0)
                    vector of I values for permutated samples
    p_sim         : float (if permutations>0)
                    p-value based on permutations
    EI_sim        : array (if permutations>0)
                    average value of I from permutations
    VI_sim        : array (if permutations>0)
                    variance of I from permutations
    seI_sim       : array (if permutations>0)
                    standard deviation of I under permutations.
    z_sim         : array (if permutations>0)
                    standardized I based on permutations
    p_z_sim       : float  (if permutations>0)
                    p-value based on standard normal approximation from
                    permutations

    Notes
    -----

    Inference is only based on permutations as analytical results are none too reliable.

    Examples
    --------
    >>> import pysal
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> f=pysal.open("../examples/sids2.dbf")
    >>> SIDR74=np.array(f.by_col['SIDR74'])
    >>> SIDR79=np.array(f.by_col['SIDR79'])
    >>> w=pysal.open("../examples/sids2.gal").read()
    >>> mbi=Moran_BV(SIDR79,SIDR74,w)
    >>> print mbi.I
    0.156131961696
    >>> mbi.p_z_sim
    0.0028373234843530604
    
    
    """
    def __init__(self,x,y,w,transformation="r",permutations=PERMUTATIONS):
        zy=(y-y.mean())/y.std(ddof=1)
        zx=(x-x.mean())/x.std(ddof=1)
        self.zx=zx
        self.zy=zy
        w.transform=transformation
        self.w=w
        self.I=self.__calc(zy)
        if permutations:
            nrp=np.random.permutation
            sim=[self.__calc(nrp(zy)) for i in xrange(permutations)]
            self.sim=sim
            self.p_sim=(sum(sim >= self.I)+1.)/(permutations+1.)
            self.EI_sim = sum(sim)/permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim=(self.I - self.EI_sim)/self.seI_sim
            self.p_z_sim=2.0*(1-stats.norm.cdf(np.abs(self.z_sim)))
          
    def __calc(self,zy):
        wzy=slag(self.w,zy)
        self.num=sum(self.zx*wzy)
        self.den=sum(zy*zy)
        return self.num/self.den

def Moran_BV_matrix(variables,w,permutations=0,varnames=None):
    """Bivariate Moran Matrix

    Calculates bivariate Moran between all pairs of a set of variables.

    Parameters
    ----------
    variables    : list 
                   sequence of variables 
    w            : W
                   a spatial weights object
    permutations : int
                   number of permutations
    varnames     : list
                   strings for variable names. If specified runtime summary is
                   printed

    Returns
    -------
    results      : dictionary
                   (i,j) is the key for the pair of variables, values are the
                   Moran_BV object.

    Examples
    --------
    >>> import pysal

    open dbf

    >>> f=pysal.open("../examples/sids2.dbf")

    pull of selected variables from dbf and create numpy arrays for each

    >>> varnames=['SIDR74','SIDR79','NWR74','NWR79']
    >>> vars=[np.array(f.by_col[var]) for var in varnames]

    create a contiguity matrix from an external gal file

    >>> w=pysal.open("../examples/sids2.gal").read()

    create an instance of Moran_BV_matrix

    >>> res=Moran_BV_matrix(vars,w,varnames=varnames)

    check values

    >>> res[(0,1)].I
    0.19362610652874668
    >>> res[(3,0)].I
    0.37701382542927858
    

    """

    k=len(variables)
    rk=range(0,k-1)
    results={}
    for i in rk:
        for j in range(i+1,k):
            y1=variables[i]
            y2=variables[j]
            results[i,j]=Moran_BV(y1,y2,w,permutations=permutations)
            results[j,i]=Moran_BV(y2,y1,w,permutations=permutations)
    return results

class Moran_Local:
    """Local Moran Statistics


    Parameters
    ----------
    y : n*1 array

    w : weight instance assumed to be aligned with y

    transformation : string
                     weights transformation, default is row-standardized "r".
                     Other options include "B": binary, "D":
                     doubly-standardized, "U": untransformed (general weights),
                     "V": variance-stabilizing.

    permutations   : number of random permutations for calculation of pseudo-p_values


    Attributes
    ----------

    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of random permutations for calculation of pseudo-p_values
    I            : float
                   value of Moran's I
    q            : array (if permutations>0)
                   values indicate quadrat location 1 HH, 2 LH, 3 LL, 4 HL 
    sim          : array (if permutations>0)
                   vector of I values for permutated samples
    p_sim        : array (if permutations>0)
                   p-value based on permutations
    EI_sim       : float (if permutations>0)
                   average value of I from permutations
    VI_sim       : float (if permutations>0)
                   variance of I from permutations
    seI_sim      : float (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float (if permutations>0)
                   p-value based on standard normal approximation from
                   permutations

    Notes 
    -----
    p-values are one sided - where side is based on the original I value
    for each observation (in self.Is). In other words extreme is considered
    being further away from the origin and in the same direction than original I
    statistic  for the focal observation.
    

    Examples
    --------
    >>> import pysal
    >>> import numpy as np
    >>> from pysal.esda import moran
    >>> np.random.seed(10)
    >>> w=pysal.open("../examples/desmith.gal").read()
    >>> f=pysal.open("../examples/desmith.txt")
    >>> y=np.array(f.by_col['z'])
    >>> lm=Moran_Local(y,w,transformation="r",permutations=99)
    >>> lm.q
    array([4, 4, 4, 2, 3, 3, 1, 4, 3, 3])
    >>> lm.p_z_sim[0]
    0.99036648060872201


    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures
    """
    def __init__(self,y,w,transformation="r",permutations=PERMUTATIONS):
        self.y=y
        n=len(y)
        self.n=n
        self.n_1=n-1
        z=y-y.mean()
        z/=y.std()
        self.z=z
        w.transform=transformation
        self.w=w
        self.permutations=permutations
        self.den=sum(z*z)
        self.Is = self.calc(self.w,self.z)
        self.__quads()
        if permutations:
            self.__crand()
            pos=self.Is>0
            neg=self.Is<=0
            sim=np.transpose(self.rlisas)
            above=sim >= self.Is
            below=sim <= self.Is
            p=pos*above + neg*below
            self.p_sim = (sum(p)+1.0)/(permutations+1)
            self.sim=sim
            self.EI_sim = sim.mean()
            self.seI_sim = sim.std()
            self.VI_sim = self.seI_sim * self.seI_sim
            self.z_sim = (self.Is-self.EI_sim)/self.seI_sim
            self.p_z_sim=2.0*(1-stats.norm.cdf(np.abs(self.z_sim)))

    def calc(self,w,z):
        zl=slag(w,z)
        return self.n_1*self.z*zl/self.den


    def __crand(self):
        """
        conditional randomization

        for observation i with ni neighbors, the candidate set cannot include
        i (we don't want i being a neighbor of i). we have to sample without
        replacement from a set of ids that doesn't include i. numpy doesn't
        directly support sampling wo replacement and it is expensive to
        implement this. instead we omit i from the original ids, permutate the
        ids and take the first ni elements of the permutated ids as the
        neighbors to i in each randomization.

        """
        z=self.z
        lisas=np.zeros((self.n,self.permutations))
        n_1=self.n-1
        rid=range(n_1)
        prange=range(self.permutations)
        rids=np.array([np.random.permutation(rid) for i in prange])
        ids=np.arange(self.w.n)
        w=self.w.weights
        wc=self.w.cardinalities
        ido=self.w.id_order
        for i in range(self.w.n):
            idsi=ids[ids!=i]
            lisas[i]=[z[i]*sum(w[ido[i]]*z[idsi[rid[0:wc[ido[i]]]]]) for rid in rids]
        self.rlisas=(n_1/self.den)*lisas
    
    def __quads(self):
        zl=slag(self.w,self.z)
        zp=self.z>0
        lp=zl>0
        pp=zp*lp
        np=(1-zp)*lp
        nn=(1-zp)*(1-lp)
        pn=zp*(1-lp)
        self.q=1*pp+2*np+3*nn+4*pn

def _test():
    import doctest
    doctest.testmod(verbose=True)
    
if __name__ == '__main__':
    _test()
