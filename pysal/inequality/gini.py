"""
Gini based Inequality Metrics
"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

from pysal.common import *
import numpy as np
__all__ = ['Gini', 'Gini_Spatial']


import pysal as ps
import numpy as np


class Gini:
    """
    Classic Gini coefficient in absolute deviation form
    """
    def __init__(self,x):

        x.shape = (x.shape[0],)
        d = np.abs(np.array([x - xi for xi in x]))
        n = len(x)
        xbar = x.mean()
        den = xbar * 2 * n**2
        dtotal = d.sum()
        return dtotal / den

class Gini_Spatial:
    """
    Spatial Gini coefficient

    Parameters
    ----------

    y: array (n,1)
      
    w: binary spatial weights object

    Examples
    --------
    >>> import pysal
    >>> f=pysal.open(pysal.examples.get_path("mexico.csv"))
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> regimes=np.array(f.by_col('hanson98'))
    >>> w = pysal.regime_weights(regimes)
    >>> gs = pysal.inequality.gini.Gini_Spatial(y[:,0],w)
    >>> gs.p_sim
    0.01

 
    >>>
 
    """
    def __init__(self,x, w, permutations = 99):
        x.shape = (x.shape[0],)
        d = np.abs(np.array([x - xi for xi in x]))
        n = len(x)
        xbar = x.mean()
        den = xbar * 2 * n**2
        wg = w.sparse.multiply(d).sum()
        self.wg = wg # spatial inequality component
        dtotal = d.sum()
        wcg = dtotal - wg # complement to spatial inequality component
        self.wcg = wcg
        self.g = dtotal / den
        self.wcg_share = wcg / dtotal
        self.dtotal = dtotal
        self.den = den

        if permutations:
            ids = np.arange(n)

            wcgp = np.zeros((permutations,1))
            for perm in xrange(permutations):
                # permute rows/cols of d
                np.random.shuffle(ids)
                wcgp[perm] = w.sparse.multiply(d[ids,:][:,ids]).sum()
            above = wcgp >= self.wcg
            larger = above.sum()
            if (permutations -  larger) <  larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.e_wg_sim = wcgp.mean()
            self.std_wg_sim =  wcgp.std()
            self.wgs_sim = wcgp
