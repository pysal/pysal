"""
Gini based Inequality Metrics
"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

#from pysal.common import *
import numpy as np
from scipy.stats import norm as NORM

__all__ = ['Gini', 'Gini_Spatial']


class Gini:
    """
    Classic Gini coefficient in absolute deviation form

    Parameters
    ----------

    y : array (n,1)
       attribute

    Attributes
    ----------

    g : float
       Gini coefficient

    """

    def __init__(self, x):

        x.shape = (x.shape[0],)
        d = np.abs(np.array([x - xi for xi in x]))
        n = len(x)
        xbar = x.mean()
        den = xbar * 2 * n**2
        dtotal = d.sum()
        self.g = dtotal/den


class Gini_Spatial:
    """
    Spatial Gini coefficient

    Provides for computationally based inference regarding the contribution of
    spatial neighbor pairs to overall inequality across a set of regions. [Rey2013]_
    
    Parameters
    ----------

    y : array (n,1)
       attribute

    w : binary spatial weights object

    permutations : int (default = 99)
       number of permutations for inference

    Attributes
    ----------

    g : float
       Gini coefficient

    wg : float
       Neighbor inequality component (geographic inequality)

    wcg : float
       Non-neighbor inequality component (geographic complement inequality)

    wcg_share : float
       Share of inequality in non-neighbor component

    If Permuations > 0

    p_sim : float
       pseudo p-value for spatial gini

    e_wcg : float
       expected value of non-neighbor inequality component (level) from permutations

    s_wcg : float
           standard deviation non-neighbor inequality component (level) from permutations

    z_wcg : float
           z-value non-neighbor inequality component (level) from permutations

    p_z_sim : float
             pseudo  p-value based on standard normal approximation of permutation based values


    Examples
    --------
    >>> import pysal
    >>> import numpy as np

    Use data from the 32 Mexican States, Decade frequency 1940-2010

    >>> f=pysal.open(pysal.examples.get_path("mexico.csv"))
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))

    Define regime neighbors

    >>> regimes=np.array(f.by_col('hanson98'))
    >>> w = pysal.block_weights(regimes)
    >>> np.random.seed(12345)
    >>> gs = pysal.inequality.gini.Gini_Spatial(y[:,0],w)
    >>> gs.p_sim
    0.01
    >>> gs.wcg
    4353856.0
    >>> gs.e_wcg
    1067629.2525252525
    >>> gs.s_wcg
    95869.167798782844
    >>> gs.z_wcg
    34.2782442252145
    >>> gs.p_z_sim
    0.0

    Thus, the amount of inequality between pairs of states that are not in the
    same regime (neighbors) is significantly higher than what is expected
    under the null of random spatial inequality.

    """
    def __init__(self, x, w, permutations=99):
        x.shape = (x.shape[0],)
        d = np.abs(np.array([x - xi for xi in x]))
        n = len(x)
        xbar = x.mean()
        den = xbar * 2 * n**2
        wg = w.sparse.multiply(d).sum()
        self.wg = wg  # spatial inequality component
        dtotal = d.sum()
        wcg = dtotal - wg  # complement to spatial inequality component
        self.wcg = wcg
        self.g = dtotal / den
        self.wcg_share = wcg / dtotal
        self.dtotal = dtotal
        self.den = den

        if permutations:
            ids = np.arange(n)
            wcgp = np.zeros((permutations, 1))
            for perm in xrange(permutations):
                # permute rows/cols of d
                np.random.shuffle(ids)
                wcgp[perm] = w.sparse.multiply(d[ids, :][:, ids]).sum()
            above = wcgp >= self.wcg
            larger = above.sum()
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.e_wcg = wcgp.mean()
            self.s_wcg = wcgp.std()
            self.z_wcg = (self.wcg - self.e_wcg) / self.s_wcg
            self.p_z_sim = 1.0 - NORM.cdf(self.z_wcg)
