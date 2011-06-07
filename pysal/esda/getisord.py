"""
Getis and Ord G statistic for spatial autocorrelation
"""
__author__ = "Sergio J. Rey <srey@asu.edu>, Myunghwa Hwang <mhwang4@gmail.com> "
__all__ = ['G']

from pysal.common import np, stats, math
from pysal.weights.spatial_lag import lag_spatial as slag

PERMUTATIONS=999

class G:
    """
    Global G Autocorrelation Statistic
    
    Parameters:
    -----------
    y: array
    w: DistanceBand W
       spatial weights based on distance band
    permutations: int
                  the number of random permutations for calculating
                  pseudo p_values

    Attributes:
    -----------
    y: array 
       original variable
    w: DistanceBand W  
       spatial weights based on distance band
    permutation: int
                 the number of permutations
    G: float 
       the value of statistic
    EG: float 
        the expected value of statistic
    VG: float
        the variance of G under normality assumption
    z_norm: float
         standard normal test statistic
    p_norm: float
            p-value under normality assumption (one-tailed)
    sim: array (if permutations > 0) 
         vector of G values for permutated samples
    p_sim: float 
           p-value based on permutations
    EG_sim: float 
            average value of G from permutations
    VG_sim: float 
            variance of G from permutations
    seG_sim: float
             standard deviation of G under permutations.
    z_sim: float 
           standardized G based on permutations
    p_z_sim: float
             p-value based on standard normal approximation from
             permutations

    Notes
    -----
    Moments are based on normality assumption. 

    Examples
    --------
    >>> from pysal.weights.Distance import DistanceBand
    >>> import numpy 
    >>> numpy.random.seed(10)

    Preparing a point data set

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points

    >>> w = DistanceBand(points,threshold=15)
    >>> w.transform = "B"

    Prepareing a variable 

    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord G test
    >>> g = G(y,w)

    Examining the results
    >>> print "%.8f" % g.G
    0.55709779

    >>> print "%.4f" % g.p_norm
    0.1729
 
    """
    def __init__(self, y, w, permutations=PERMUTATIONS):
        self.n = len(y)
        self.y = y
        w.transform = "B"
        self.w = w
        self.permutations = permutations
        self.__moments()
        xn = xrange(len(y))
        self.xn = xn
        self.y2 = y*y
        self.den = [y[i]*y[j] for i in xn for j in xn if i!=j ]
        self.G = self.__calc(self.y)
        self.z_norm = (self.G - self.EG) / math.sqrt(self.VG)
        self.p_norm = 1 - stats.norm.cdf(np.abs(self.z_norm))
        
        if permutations:
            sim = [self.__calc(np.random.permutation(self.y)) \
                 for i in xrange(permutations)]
            self.sim = sim
            self.p_sim = (sum(sim>=self.G) + 1.) / (permutations + 1.)
            self.EG_sim = sum(sim) / permutations
            self.seG_sim = np.array(sim).std()
            self.VG_sim = self.seG_sim**2
            self.z_sim = (self.G - self.EG_sim) / self.seG_sim
            self.p_z_sim= 1.-stats.norm.cdf(np.abs(self.z_sim))

    def __moments(self):
        y = self.y
        n = self.n
        w = self.w
        n2 = n*n
        s0 = w.s0
        self.EG = s0 / (n*(n-1))
        s02 = s0*s0
        s1 = w.s1
        s2 = w.s2
        b0 = (n2 - 3*n + 3)*s1 - n*s2 + 3*s02
        b1 = (-1.)*((n2 - n)*s1 - 2*n*s2 + 6*s02)
        b2 = (-1.)*(2*n*s1 - (n+3)*s2 + 6*s02)
        b3 = 4*(n-1)*s1 - 2*(n+1)*s2 + 8*s02
        b4 = s1 - s2 + s02
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        y2 = y*y
        y3 = y*y2
        y4 = y2*y2
        EG2 = (b0*(sum(y2)**2) + b1*sum(y4) + b2*(sum(y)**2)*sum(y2))
        EG2 += b3*sum(y)*sum(y3) + b4*(sum(y)**4)
        EG2NUM = EG2
        EG2DEN = (((sum(y)**2 - sum(y2))**2)*n*(n-1)*(n-2)*(n-3))
        self.EG2 = EG2NUM / EG2DEN
        self.VG = self.EG2 - self.EG**2

    def __calc(self, y):
        yl = slag(self.w,y)
        self.num = y*yl
        return sum(self.num) / sum(self.den)

class G_Local:
    """ 
    Generalized Local G Autocorrelation Statistic
    
    Parameters:
    -----------
    y: array
       variable
    w: DistanceBand W
       weights instance that is based on threshold distance and assumed to be aligned with y
    permutations: int
                  the number of random permutations for calculating
                  pseudo-p_values
    star: boolean
          whether or not to include focal observation in sums
          default is False 
    diag_wgt: scalar or numpy array
              a set of numerical values representing weights for observations themselves
              default is 1.0

    Attributes:
    -----------
    y: array
       original variable
    w: DistanceBand W
       original weights object
    permutations: int
                 the number of permutations
    Gs: array of floats
        the value of statistic
        G and G* at an observation i follow N(0,1),
        meaning that they are standard variates
    p_norm: array of floats
            p-value under normality assumption (two-sided)
            such as Gi ~ N(0,1) and G*i ~ N(0,1)
    sim: array of arrays of floats (if permutations>0)
         vector of I values for permutated samples
    p_sim: array of floats
           p-value based on permutations
    EG_sim: array of floats 
            average value of G from permutations
    VG_sim: array of floats
            variance of G from permutations
    seG_sim: array of floats
             standard deviation of G under permutations.
    z_sim: array of floats
           standardized G based on permutations
    p_z_sim: array of floats 
             p-value based on standard normal approximation from
             permutations

    Notes
    -----
    Moments are based on normality assumption and when star is False. 
    If star is true, inference should be based on permutations and not the
    normal approximation. Under the null hypothesis, it is known that 
    the expectation and variance are 0 and 1, since G and G* can be 
    considered standard variates (see Getis and Ord, 1996). 

    References
    ----------
    Ord, J.K. and Getis, A. (1995) Local spatial autocorrelation statistics: 
    distributional issues and an application. Geographical Analysis, 27(4):286-306
    Getis, A. and Ord, J. K. (1996) Local spatial statistics: an overview,
    in Spatial Analysis: Modelling in a GIS Environment, edited by Longley, P. 
    and Batty, M.

    Examples
    --------
    >>> from pysal.weights.Distance import DistanceBand
    >>> import numpy 
    >>> numpy.random.seed(10)

    Preparing a point data set

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points

    >>> w = DistanceBand(points,threshold=15)
    >>> w.transform = "B"

    Prepareing a variable 

    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord local G test
    >>> lg = G_Local(y,w)

    Examining the results
    >>> lg.Gs
    array([-1.0136729 , -0.04361589,  1.31558703, -0.31412676,  1.15373986,
            1.77833941])
    >>> lg.p_z_sim[0]
    0.30986577755682521
        
    >>> numpy.random.seed(10)

    Applying Getis and Ord local G* test
    >>> lg_star = G_Local(y,w, star=True, diag_wgt=1.0)

    Examining the results
    >>> lg_star.Gs
    array([-1.39727626, -0.28917762,  0.65064964, -0.28917762,  1.23452088,
            2.02424331])
    >>> lg_star.p_z_sim[0]
    0.93819450404703719

    """
    def __init__(self, y, w, permutations=PERMUTATIONS, star=False, diag_wgt=1.0):
        self.n = len(y)
        self.y = y
        self.w = w
        self.permutations = permutations
        self.star = star
        self.diag_wgt = diag_wgt
        self.Gs = self.calc()
        self.p_norm = np.array([2.0*(1 - stats.norm.cdf(np.abs(i))) for i in self.Gs])
        if permutations:
            self.__crand()
            pos = self.Gs > 0
            neg = self.Gs <= 0
            sim = np.transpose(self.rGs)
            above = sim >= self.Gs
            below = sim <= self.Gs
            p = pos*above + neg*below
            self.p_sim = (sum(p) + 1.0)/(permutations + 1)
            self.sim = sim
            self.EG_sim = sim.mean()
            self.seG_sim = sim.std()
            self.VG_sim = self.seG_sim * self.seG_sim
            self.z_sim = (self.Gs - self.EG_sim)/self.seG_sim
            self.p_z_sim = 2.0*(1-stats.norm.cdf(np.abs(self.z_sim)))

    def __crand(self):
        y = self.y
        rGs = np.zeros((self.n, self.permutations))
        n_1 = self.n - 1
        rid = range(n_1)
        prange = range(self.permutations)
        k = self.w.max_neighbors + 1
        rids = np.array([np.random.permutation(rid)[0:k] for i in prange])
        ids = np.arange(self.w.n)
        ido = self.w.id_order
        w = [self.w.weights[ido[i]] for i in ids]
        wc = [self.w.cardinalities[ido[i]] for i in ids]
        star, dwgt = self.star, self.diag_wgt
        if not isinstance(dwgt, np.ndarray):
            dwgt = [dwgt]*self.w.n
        Wybar, den = self.W_y_mean, self.denum
        for i in range(self.w.n):
            idsi = ids[ids!=i]
            np.random.shuffle(idsi)
            w_i, i_val = w[i], star*dwgt[i]
            Wybar, den = self.W_y_mean[i], self.denum[i]
            rGs[i] = [((sum(w_i*y[idsi[rid[0:wc[i]]]]) + i_val) - Wybar)/den for rid in rids]
        self.rGs = rGs

    def calc(self):
        y = self.y
        yl = slag(self.w, y)
        ido = self.w.id_order
        W = np.zeros(self.n)
        S1 = np.zeros(self.n)
        for i in range(self.n):
            wgt = np.array(self.w.weights[ido[i]])
            W[i] = wgt.sum()
            S1[i] = (wgt*wgt).sum()

        if not self.star:
            N = self.n - 1
            y2 = y*y
            y_sum = sum(y)
            y2_sum = sum(y2)
            y_mean = (y_sum - y)/N*1.0
            y2_mean = (y2_sum - y2)/N*1.0
            s = np.sqrt(y2_mean - y_mean*y_mean)
        else:
            N = self.n
            s = y.std()
            y_mean = y.mean()
            W += self.diag_wgt
            S1 = S1 + self.diag_wgt*self.diag_wgt
            yl = yl + y

        num = yl - W*y_mean
        self.W_y_mean = W*y_mean
        self.denum = s*np.sqrt((N*S1 - W*W)/(N-1))

        return num/self.denum

def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
