"""
Geary's C statistic for spatial autocorrelation


Author(s):
    Serge Rey srey@asu.edu



"""
from pysal.common import *

PERMUTATIONS=999

class Geary:
    """Global Geary C Autocorrelation statistic
    
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

        C: value of statistic

        EC: expected value
        
        VC: variance of G under normality assumption

        z_norm: z-statistic for C under normality assumption

        z_rand: z-statistic for C under randomization assumption

        p_norm: p-value under normality assumption (one-tailed)

        p_rand: p-value under randomization assumption (one-tailed)


        (if permutations>0)
        sim: vector of I values for permutated samples

        p_sim: p-value based on permutations

        EC_sim: average value of C from permutations

        VC_sim: variance of C from permutations

        seC_sim: standard deviation of C under permutations.

        z_sim: standardized C based on permutations

        p_z_sim: p-value based on standard normal approximation from
        permutations

    Example:
        >>> import pysal
        >>> w=pysal.open("../examples/book.gal").read()
        >>> f=pysal.open("../examples/book.txt")
        >>> y=np.array(f.by_col['y'])
        >>> c=Geary(y,w,permutations=0)
        >>> c.C
        0.33281733746130032
        >>> c.p_norm
        0.00076052983736881971
        >>> 
    """
    def __init__(self,y,w,transformation="B",permutations=PERMUTATIONS):
        self.n=len(y)
        self.y=y
        w.transform=transformation
        self.w=w
        self.permutations=permutations
        self.__moments()
        xn=xrange(len(y))
        self.xn=xn
        self.y2=y*y
        yd=y-y.mean()
        yss=sum(yd*yd)
        self.den = yss*self.w.s0 * 2.0
        self.C=self.__calc(y)
        de=self.C-1.0
        self.EC=1.0
        self.z_norm=de/self.seC_norm
        self.z_rand=de/self.seC_rand
        self.p_norm = stats.norm.pdf(self.z_norm)
        self.p_rand = stats.norm.pdf(self.z_rand)


        
        if permutations:
            sim=[self.__calc(np.random.permutation(self.y)) \
                 for i in xrange(permutations)]
            self.sim=sim
            self.p_sim =(sum(sim>=self.C)+1)/(permutations+1.)
            self.EC_sim=sum(sim)/permutations
            self.seC_sim=np.array(sim).std()
            self.VC_sim=self.seC_sim**2
            self.z_sim = (self.C - self.EC_sim)/self.seC_sim
            self.p_z_sim = stats.norm.pdf(self.z_sim)

    def __moments(self):
        y=self.y
        n=self.n
        w=self.w
        s0=w.s0
        s1=w.s1
        s2=w.s2
        s02=s0*s0

        yd=y-y.mean()
        k = (1/(sum(yd**4)) * ((sum(yd**2))**2))
        vc_rand = (1/(n*((n-2)**2)*s02)) * ((((n-1)*s1) * (n*n-3*n+3-(n-1)*k)) \
             - ((.25*(n-1)*s2) * (n*n+3*n-6-(n*n-n+2)*k)) \
                + (s02* (n*n-3-((n-1)**2)*k)))
        vc_norm = ((1 / (2 * (n+1) * s02)) * ((2*s1+s2) * (n-1) - 4 * s02))

        self.VC_rand = vc_rand
        self.VC_norm = vc_norm
        self.seC_rand = vc_rand**(0.5)
        self.seC_norm = vc_norm**(0.5)

    
    def __calc(self,y):
        ys=np.zeros(y.shape)
        y2=y**2
        for i in self.w.weights:
            neighbors=self.w.neighbors[i]
            wijs=self.w.weights[i]
            z=zip(neighbors,wijs)
            ys[i] = sum([wij*(y2[i] - 2*y[i]*y[j] + y2[j]) for j,wij in z])
        a= (self.n-1)* sum(ys)
        return a/self.den

    
class __TestC(unittest.TestCase):

    def setUp(self):
        pass

    def test_C(self):
        import pysal
        w=pysal.open("../examples/book.gal").read()
        f=pysal.open("../examples/book.txt")
        y=np.array(f.by_col['y'])
        c=Geary(y,w,permutations=0)
        v="%6.5f"%c.C
        self.assertEquals(v,'0.33282')



if __name__ == '__main__':
    import unittest
    import doctest
    import geary
    suite = unittest.TestSuite()
    for mod in [geary]:
        suite.addTest(doctest.DocTestSuite(mod))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    # regular unittest
    unittest.main()


