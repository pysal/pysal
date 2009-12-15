"""
Moran's I Spatial Autocorrelation Statistics


Author(s):
    Serge Rey srey@asu.edu

"""
from pysal.common import *

from spatial_lag import lag as slag

PERMUTATIONS=999

class Moran:
    """Moran's I Global Autocorrelation Statistic
    
    Arguments:
        y: n*1 array

        w: weight instance assumed to be aligned with y

        transformation: weights transformation, default is row-standardized
        "W". Other options include "B": binary, "D": doubly-standardized, "U":
            untransformed (general weights), "V": variance-stabilizing.

        permutations: number of random permutations for calculation of
        pseudo-p_values


    Attributes:
        y: original variable

        w: original w object

        permutation: number of permutations

        I: value of Moran's I

        EI: expected value under normality assumption

        VI_norm: variance of I under normality assumption

        seI_norm: standard deviation of I under normality assumption

        z_norm: z-value of I under normality assumption

        p_norm: p-value of I under normality assumption (1-tailed)

        VI_rand: variance of I under randomization assumption

        seI_rand: standard deviation of I under randomization assumption

        z_rand: z-value of I under randomization assumption

        p_rand: p-value of I under randomization assumption (1-tailed)

        (if permutations>0)
        sim: vector of I values for permutated samples

        p_sim: p-value based on permutations

        EI_sim: average value of I from permutations

        VI_sim: variance of I from permutations

        seI_sim: standard deviation of I under permutations.

        z_sim: standardized I based on permutations

        p_z_sim: p-value based on standard normal approximation from
        permutations

    Example:
        >>> import pysal
        >>> w=pysal.open("../examples/stl.gal").read()
        >>> w.id_order=range(78)
        >>> f=pysal.open("../examples/stl_hom.txt")
        >>> y=np.array(f.by_col['HR8893'])
        >>> mi=Moran(y,w)
        >>> mi.I
        0.24365582621771659
        >>> mi.EI
        -0.012987012987012988
        >>> mi.p_norm
        0.00052730423329256173
        >>> 
        
    """
    def __init__(self,y,w,transformation="W",permutations=PERMUTATIONS):
        self.y=y
        w.transform=transformation
        self.w=w
        self.permutations=permutations
        self.__moments()
        self.I = self.__calc(self.z)
        self.z_norm = (self.I - self.EI)/self.seI_norm
        self.p_norm = stats.norm.pdf(self.z_norm)
        self.z_rand = (self.I - self.EI)/self.seI_rand
        self.p_rand = stats.norm.pdf(self.z_rand)


        if permutations:
            sim=[self.__calc(np.random.permutation(self.z)) \
                 for i in xrange(permutations)]
            self.sim=sim
            self.p_sim=(sum(sim >= self.I)+1.)/(permutations+1.)
            self.EI_sim = sum(sim)/permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim=(self.I - self.EI_sim)/self.seI_sim
            self.p_z_sim=stats.norm.pdf(self.z_sim)

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
        zl=slag(z,self.w)
        self.inum=sum(self.z*zl)
        return self.n/self.w.s0 * sum(self.z*zl)/self.z2ss

class Moran_BV:
    """Bivariate Moran's I
    
 
    
    Arguments:
        y1: n*1 array

        y2: n*1 array 

        w: weight instance assumed to be aligned with y

        transformation: weights transformation, default is row-standardized
        "W". Other options include "B": binary, "D": doubly-standardized, "U":
            untransformed (general weights), "V": variance-stabilizing.

        permutations: number of random permutations for calculation of
        pseudo-p_values


    Attributes:
        y: original variable

        w: original w object

        permutation: number of permutations

        I: value of Moran's I


        (if permutations>0)
        sim: vector of I values for permutated samples

        p_sim: p-value based on permutations

        EI_sim: average value of I from permutations

        VI_sim: variance of I from permutations

        seI_sim: standard deviation of I under permutations.

        z_sim: standardized I based on permutations

        p_z_sim: p-value based on standard normal approximation from
        permutations


    Notes:
        Inference is only based on permutations as analytical results are
        none too reliable.
    """
    def __init__(self,y1,y2,w,transformation="W",permutations=PERMUTATIONS):
        self.y1=y1
        self.y2=y2
        self.w=w
        z1=y1-y1.mean()
        z1/=y1.std()
        self.z1=z1
        self.z12ss=sum(z1*z1)
        z2=y2-y2.mean()
        z2/=y2.std()
        self.I=self.__calc(z2)
        if permutations:
            nrp=np.random.permutation
            sim=[self.__calc(nrp(z2)) for i in xrange(permutations)]
            self.sim=sim
            self.p_sim=(sum(sim >= self.I)+1.)/(permutations+1.)
            self.EI_sim = sum(sim)/permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim**2
            self.z_sim=(self.I - self.EI_sim)/self.seI_sim
            self.p_z_sim=stats.norm.pdf(self.z_sim)


    def __calc(self,z2):
        z2l=slag(z2,self.w)
        self.inum=sum(self.z1,z2l)
        return sum(self.z1*z2l)/self.z12ss



def Moran_BV_matrix(variables,w,permutations=0):
    """Bivariate Moran Matrix

    Calculates bivariate Moran between all pairs of a set of variables.

    Arguments:
        variables: a list of variables (arrays)

        w: a spatial weights object

        permutations: number of permutations

    Returns:
        results: a dictionary with the key having the ids for the pair of
        variables, values are the Moran_BV object.

    """

    k=len(variables)
    rk=range(0,k-1)
    results={}
    for i in rk:
        for j in range(i+1,k):
            y1=variables[i]
            y2=variables[j]
            results[i,j]=Moran_BV(y1,y2,w,permutations)
            results[j,i]=Moran_BV(y2,y1,w,permutations)
    return results


class Moran_Local:
    """Local Moran Statistics


    Arguments:
        y: n*1 array

        w: weight instance assumed to be aligned with y

        transformation: weights transformation, default is row-standardized
        "W". Other options include "B": binary, "D": doubly-standardized, "U":
            untransformed (general weights), "V": variance-stabilizing.

        permutations: number of random permutations for calculation of
        pseudo-p_values


    Attributes:
        y: original variable

        w: original w object

        permutation: number of permutations

        Is: values of Moran's I

        q: array of values indicated quadrat location: 1 HH, 2 LH, 3 LL, 4 HL

        (if permutations>0)
        sim: vector of I values for permutated samples

        p_sim: p-value based on permutations

        EI_sim: average values of I from permutations

        VI_sim: variance of I from permutations

        seI_sim: standard deviation of I under permutations.

        z_sim: standardized I based on permutations

        p_z_sim: p-value based on standard normal approximation from
        permutations.
        
        
        Note: p-values are one sided - where side is based on the
        original I value for each observation (in self.Is). In other words
        extreme is considered being further away from the origin and in the
        same direction than original I statistic  for the focal observation.

    Example
    -------
        >>> import pysal
        >>> w=pysal.open("../examples/desmith.gal").read()
        >>> w.id_order=range(w.n)
        >>> f=pysal.open("../examples/desmith.txt")
        >>> y=np.array(f.by_col['z'])
        >>> lm=Moran_Local(y,w,transformation="W",permutations=0)
        >>> lm.q
        array([4, 4, 4, 2, 3, 3, 1, 4, 3, 3])
        >>> lm.Is
        array([-0.11409277, -0.19940543, -0.13351408, -0.51770383,  0.48095009,
                0.12208113,  1.19148298, -0.58144305,  0.07101383,  0.34314301])
    """
    def __init__(self,y,w,transformation="W",permutations=PERMUTATIONS):
        self.y=y
        z=y-y.mean()
        z/=y.std()
        self.z=z
        w.transform=transformation
        self.w=w
        self.permutations=permutations
        self.den=sum(z*z)
        self.Is = self.__calc(self.z)
        self.__quads()
        n=len(y)
        if permutations:
            sim=[self.__calc(np.random.permutation(self.z)) \
                 for i in xrange(permutations)]
            self.sim=sim
            pos=self.Is>0
            neg=self.Is<=0
            sim=np.array(sim)
            above=sim >= self.Is
            below=sim <= self.Is
            p=pos*above + neg*below
            self.p_sim = (sum(p)+1.0)/(permutations+1)
            self.sim=sim
            self.EI_sim = sim.mean()
            self.seI_sim = sim.std()
            self.VI_sim = self.seI_sim * self.seI_sim
            self.z_sim = (self.Is-self.EI_sim)/self.seI_sim
            self.p_z_sim=stats.norm.pdf(self.z_sim)

    def __calc(self,z):
        zl=slag(z,self.w)
        num=self.z * zl
        return (len(zl)-1)*self.z*zl/self.den
    
    def __quads(self):
        zl=slag(self.z,self.w)
        zp=self.z>0
        lp=zl>0
        pp=zp*lp
        np=(1-zp)*lp
        nn=(1-zp)*(1-lp)
        pn=zp*(1-lp)
        self.q=1*pp+2*np+3*nn+4*pn


# tests
class __TestMoran(unittest.TestCase):

    def setUp(self):
        import pysal
        self.w=pysal.open("../examples/stl.gal").read()
        self.w.id_order=range(78)
        f=pysal.open("../examples/stl_hom.txt")
        self.y1=np.array(f.by_col['HR8893'])
        self.y2=np.array(f.by_col['HR8488'])

    def test_I(self):
        mi=Moran(self.y2,self.w)
        f="%6.4f"%mi.I
        self.assertEquals(f,"0.2068")

    def test_local(self):
        import pysal
        w=pysal.open("../examples/desmith.gal").read()
        w.id_order=range(w.n)
        f=pysal.open("../examples/desmith.txt")
        y=np.array(f.by_col['z'])
        lm=Moran_Local(y,w,transformation="W",permutations=0)
        v="%6.4f"%lm.Is[2]
        self.assertEquals(v,"-0.1335")
        q=lm.q.tolist()
        self.assertEquals(q,[4, 4, 4, 2, 3, 3, 1, 4, 3, 3])

    def test_mvI(self):
        mv=Moran_BV(self.y1,self.y2,self.w)
        v="%6.4f"%mv.I
        self.assertEquals(v,'0.9322')

if __name__ == '__main__':
    import doctest
    doctest.testmod()
