"""
Gini based Inequality Metrics
"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

import numpy as np
from scipy.stats import norm as NORM

__all__ = ['Gini', 'Gini_Spatial']


def _gini(x):
    """
    Memory efficient calculation of Gini coefficient in relative mean difference form

    Parameters
    ----------

    x : array-like

    Attributes
    ----------

    g : float
        Gini coefficient

    Notes
    -----
    Based on http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    """
    n = len(x)
    try:
        x_sum = x.sum()
    except AttributeError:
        x = np.asarray(x)
        x_sum = x.sum()
    n_x_sum = n * x_sum
    r_x = (2. * np.arange(1, len(x)+1) * x[np.argsort(x)]).sum()
    return (r_x - n_x_sum - x_sum) / n_x_sum


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

        self.g = _gini(x)


class Gini_Spatial:
    """
    Spatial Gini coefficient

    Provides for computationally based inference regarding the contribution of
    spatial neighbor pairs to overall inequality across a set of regions. See :cite:`Rey_2013_sea`.

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
    >>> import libpysal
    >>> import numpy as np
    >>> from inequality.gini import Gini_Spatial

    Use data from the 32 Mexican States, Decade frequency 1940-2010

    >>> f=libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))

    Define regime neighbors

    >>> regimes=np.array(f.by_col('hanson98'))
    >>> w = libpysal.weights.block_weights(regimes)
    >>> np.random.seed(12345)
    >>> gs = Gini_Spatial(y[:,0],w)
    >>> gs.p_sim
    0.04
    >>> gs.wcg
    4353856.0
    >>> gs.e_wcg
    4170356.7474747472

    Thus, the amount of inequality between pairs of states that are not in the
    same regime (neighbors) is significantly higher than what is expected
    under the null of random spatial inequality.

    """
    def __init__(self, x, w, permutations=99):

        x = np.asarray(x)
        g = _gini(x)
        self.g = g
        n = len(x)
        den = x.mean() * 2 * n**2
        d = g * den
        wg = self._calc(x, w)
        wcg = d - wg
        self.g = g
        self.wcg = wcg
        self.wg = wg
        self.dtotal = d
        self.den = den
        self.wcg_share = wcg / den

        if permutations:
            ids = np.arange(n)
            wcgp = np.zeros((permutations, ))
            for perm in range(permutations):
                np.random.shuffle(ids)
                wcgp[perm] = d - self._calc(x[ids], w)
            above = wcgp >= self.wcg
            larger = above.sum()
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.wcgp = wcgp
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.e_wcg = wcgp.mean()
            self.s_wcg = wcgp.std()
            self.z_wcg = (self.wcg - self.e_wcg) / self.s_wcg
            self.p_z_sim = 1.0 - NORM.cdf(self.z_wcg)

    def _calc(self, x, w):
        sad_sum = 0.0
        for i, js in w.neighbors.items():
            sad_sum += np.abs(x[i]-x[js]).sum()
        return sad_sum
