# -*- coding: utf-8 -*-
"""
Rank and spatial rank mobility measures.
"""
__author__ = "Sergio J. Rey <srey@asu.edu> "

__all__ = ['SpatialTau', 'Tau', 'Theta']

#from pysal.common import *
from scipy.stats.mstats import rankdata
from scipy.special import erfc
import pysal
import numpy as np
import scipy as sp
from numpy.random import permutation as NRP

class Theta:
    """
    Regime mobility measure. [Rey2004a]_

    For sequence of time periods Theta measures the extent to which rank
    changes for a variable measured over n locations are in the same direction
    within mutually exclusive and exhaustive partitions (regimes) of the n locations.

    Theta is defined as the sum of the absolute sum of rank changes within
    the regimes over the sum of all absolute rank changes.

    Parameters
    ----------
    y            : array 
                   (n, k) with k>=2, successive columns of y are later moments 
                   in time (years, months, etc).
    regime       : array 
                   (n, ), values corresponding to which regime each observation 
                   belongs to.
    permutations : int
                   number of random spatial permutations to generate for
                   computationally based inference.

    Attributes
    ----------
    ranks        : array
                   ranks of the original y array (by columns).
    regimes      : array
                   the original regimes array.
    total        : array 
                   (k-1, ), the total number of rank changes for each of the 
                   k periods.
    max_total    : int
                   the theoretical maximum number of rank changes for n
                   observations.
    theta        : array 
                   (k-1,), the theta statistic for each of the k-1 intervals.
    permutations : int
                   the number of permutations.
    pvalue_left  : float
                   p-value for test that observed theta is significantly lower
                   than its expectation under complete spatial randomness.
    pvalue_right : float
                   p-value for test that observed theta is significantly
                   greater than its expectation under complete spatial randomness.
                   
    Examples
    --------
    >>> import pysal
    >>> f=pysal.open(pysal.examples.get_path("mexico.csv"))
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> regime=np.array(f.by_col['esquivel99'])
    >>> np.random.seed(10)
    >>> t=Theta(y,regime,999)
    >>> t.theta
    array([[ 0.41538462,  0.28070175,  0.61363636,  0.62222222,  0.33333333,
             0.47222222]])
    >>> t.pvalue_left
    array([ 0.307,  0.077,  0.823,  0.552,  0.045,  0.735])
    >>> t.total
    array([ 130.,  114.,   88.,   90.,   90.,   72.])
    >>> t.max_total
    512

    """
    def __init__(self, y, regime, permutations=999):
        ranks = rankdata(y, axis=0)
        self.ranks = ranks
        n, k = y.shape
        ranks_d = ranks[:, range(1, k)] - ranks[:, range(k - 1)]
        self.ranks_d = ranks_d
        regimes = sp.unique(regime)
        self.regimes = regimes
        self.total = sum(abs(ranks_d))
        self.max_total = sum([abs(i - n + i - 1) for i in range(1, n + 1)])
        self._calc(regime)
        self.theta = self._calc(regime)
        self.permutations = permutations
        if permutations:
            np.perm = np.random.permutation
            sim = np.array([self._calc(
                np.perm(regime)) for i in xrange(permutations)])
            self.theta.shape = (1, len(self.theta))
            sim = np.concatenate((self.theta, sim))
            self.sim = sim
            den = permutations + 1.
            self.pvalue_left = (sim <= sim[0]).sum(axis=0) / den
            self.pvalue_right = (sim > sim[0]).sum(axis=0) / den
            self.z = (sim[0] - sim.mean(axis=0)) / sim.std(axis=0)

    def _calc(self, regime):
        within = [abs(
            sum(self.ranks_d[regime == reg])) for reg in self.regimes]
        return np.array(sum(within) / self.total)


class Tau:
    """
    Kendall's Tau is based on a comparison of the number of pairs of n
    observations that have concordant ranks between two variables.

    Parameters
    ----------
    x            : array 
                   (n, ), first variable.
    y            : array 
                   (n, ), second variable.

    Attributes
    ----------
    tau          : float
                   The classic Tau statistic.
    tau_p        : float
                   asymptotic p-value.

    Notes
    -----
    Modification of algorithm suggested by Christensen (2005). [Christensen2005]_
    PySAL implementation uses a list based representation of a binary tree for
    the accumulation of the concordance measures. Ties are handled by this
    implementation (in other words, if there are ties in either x, or y, or
    both, the calculation returns Tau_b, if no ties classic Tau is returned.)

    Examples
    --------
    # from scipy example

    >>> from scipy.stats import kendalltau
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> kt = Tau(x1,x2)
    >>> kt.tau
    -0.47140452079103173
    >>> kt.tau_p
    0.24821309157521476
    >>> skt = kendalltau(x1,x2)
    >>> skt
    (-0.47140452079103173, 0.24821309157521476)

    """

    def __init__(self, x, y):
        res = self._calc(x, y)
        self.tau = res[0]
        self.tau_p = res[1]
        self.concordant = res[2]
        self.discordant = res[3]
        self.extraX = res[4]
        self.extraY = res[5]

    def _calc(self, x, y):
        """
        List based implementation of binary tree algorithm for concordance
        measure after Christensen (2005).

        """
        x = np.array(x)
        y = np.array(y)
        n = len(y)
        perm = range(n)
        perm.sort(key=lambda a: (x[a], y[a]))
        vals = y[perm]
        ExtraY = 0
        ExtraX = 0
        ACount = 0
        BCount = 0
        CCount = 0
        DCount = 0
        ECount = 0
        DCount = 0
        Concordant = 0
        Discordant = 0
        # ids for left child
        li = [None] * (n - 1)
        # ids for right child
        ri = [None] * (n - 1)
        # number of left descendants for a node
        ld = np.zeros(n)
        # number of values equal to value i
        nequal = np.zeros(n)

        for i in range(1, n):
            NumBefore = 0
            NumEqual = 1
            root = 0
            x0 = x[perm[i - 1]]
            y0 = y[perm[i - 1]]
            x1 = x[perm[i]]
            y1 = y[perm[i]]
            if x0 != x1:
                DCount = 0
                ECount = 1
            else:
                if y0 == y1:
                    ECount += 1
                else:
                    DCount += ECount
                    ECount = 1
            root = 0
            inserting = True
            while inserting:
                current = y[perm[i]]
                if current > y[perm[root]]:
                    # right branch
                    NumBefore += 1 + ld[root] + nequal[root]
                    if ri[root] is None:
                        # insert as right child to root
                        ri[root] = i
                        inserting = False
                    else:
                        root = ri[root]
                elif current < y[perm[root]]:
                    # increment number of left descendants
                    ld[root] += 1
                    if li[root] is None:
                        # insert as left child to root
                        li[root] = i
                        inserting = False
                    else:
                        root = li[root]
                elif current == y[perm[root]]:
                    NumBefore += ld[root]
                    NumEqual += nequal[root] + 1
                    nequal[root] += 1
                    inserting = False

            ACount = NumBefore - DCount
            BCount = NumEqual - ECount
            CCount = i - (ACount + BCount + DCount + ECount - 1)
            ExtraY += DCount
            ExtraX += BCount
            Concordant += ACount
            Discordant += CCount

        cd = Concordant + Discordant
        num = Concordant - Discordant
        tau = num / np.sqrt((cd + ExtraX) * (cd + ExtraY))
        v = (4. * n + 10) / (9. * n * (n - 1))
        z = tau / np.sqrt(v)
        pval = erfc(np.abs(z) / 1.4142136)  # follow scipy
        return tau, pval, Concordant, Discordant, ExtraX, ExtraY


class SpatialTau(object):
    """
    Spatial version of Kendall's rank correlation statistic.

    Kendall's Tau is based on a comparison of the number of pairs of n
    observations that have concordant ranks between two variables. The spatial
    Tau decomposes these pairs into those that are spatial neighbors and those
    that are not, and examines whether the rank correlation is different
    between the two sets relative to what would be expected under spatial randomness.

    Parameters
    ----------
    x             : array 
                    (n, ), first variable.
    y             : array 
                    (n, ), second variable.
    w             : W
                    spatial weights object.
    permutations  : int
                    number of random spatial permutations for computationally
                    based inference.

    Attributes
    ----------
    tau                : float
                         The classic Tau statistic.
    tau_spatial        : float
                         Value of Tau for pairs that are spatial neighbors.
    taus               : array 
                         (permtuations, 1), values of simulated tau_spatial values 
                         under random spatial permutations in both periods. (Same 
                         permutation used for start and ending period).
    pairs_spatial      : int
                         Number of spatial pairs.
    concordant         : float
                         Number of concordant pairs.
    concordant_spatial : float
                         Number of concordant pairs that are spatial neighbors.
    extraX             : float
                         Number of extra X pairs.
    extraY             : float
                         Number of extra Y pairs.
    discordant         : float
                         Number of discordant pairs.
    discordant_spatial : float
                         Number of discordant pairs that are spatial neighbors.
    taus               : float
                         spatial tau values for permuted samples (if permutations>0).
    tau_spatial_psim   : float
                         pseudo p-value for observed tau_spatial under the null 
                         of spatial randomness (if permutations>0).

    Notes
    -----
    Algorithm has two stages. The first calculates classic Tau using a list
    based implementation of the algorithm from Christensen
    (2005) [Christensen2005]_. Second
    stage calculates concordance measures for neighboring pairs of locations
    using a modification of the algorithm from Press et al (2007) [Press2007]_. See Rey
    (2014) [Rey2014]_ for details.

    Examples
    --------
    >>> import pysal 
    >>> import numpy as np
    >>> f=pysal.open(pysal.examples.get_path("mexico.csv"))
    >>> vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
    >>> y=np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> regime=np.array(f.by_col['esquivel99'])
    >>> w=pysal.weights.block_weights(regime)
    >>> np.random.seed(12345)
    >>> res=[pysal.SpatialTau(y[:,i],y[:,i+1],w,99) for i in range(6)]
    >>> for r in res:
    ...     ev = r.taus.mean()
    ...     "%8.3f %8.3f %8.3f"%(r.tau_spatial, ev, r.tau_spatial_psim)
    ...
    '   0.397    0.659    0.010'
    '   0.492    0.706    0.010'
    '   0.651    0.772    0.020'
    '   0.714    0.752    0.210'
    '   0.683    0.705    0.270'
    '   0.810    0.819    0.280'
    """

    def __init__(self, x, y, w, permutations=0):

        w.transform = 'b'
        self.n = len(x)
        res = Tau(x, y)
        self.tau = res.tau
        self.tau_p = res.tau_p
        self.concordant = res.concordant
        self.discordant = res.discordant
        self.extraX = res.extraX
        self.extraY = res.extraY
        res = self._calc(x, y, w)
        self.tau_spatial = res[0]
        self.pairs_spatial = int(w.s0 / 2.)
        self.concordant_spatial = res[1]
        self.discordant_spatial = res[2]

        if permutations > 0:
            taus = np.zeros(permutations)
            ids = np.arange(self.n)
            for r in xrange(permutations):
                rids = np.random.permutation(ids)
                taus[r] = self._calc(x[rids], y[rids], w)[0]
            self.taus = taus
            self.tau_spatial_psim = pseudop(taus, self.tau_spatial,
                                            permutations)

    def _calc(self, x, y, w):
        n1 = n2 = iS = gc = 0
        ijs = {}
        for i in w.id_order:
            xi = x[i]
            yi = y[i]
            for j in w.neighbors[i]:
                if i < j:
                    ijs[(i, j)] = (i, j)
                    xj = x[j]
                    yj = y[j]
                    dx = xi - xj
                    dy = yi - yj
                    dxdy = dx * dy
                    if dxdy != 0:
                        n1 += 1
                        n2 += 1
                        if dxdy > 0.0:
                            gc += 1
                            iS += 1
                        else:
                            iS -= 1
                    else:
                        if dx != 0.0:
                            n1 += 1
                        if dy != 0.0:
                            n2 += 1
        tau_g = iS / (np.sqrt(n1) * np.sqrt(n2))
        gd = gc - iS
        return [tau_g, gc, gd]


def pseudop(sim, observed, nperm):
    above = sim >= observed
    larger = above.sum()
    psim = (larger + 1.) / (nperm + 1.)
    if psim > 0.5:
        psim = (nperm - larger + 1.) / (nperm + 1.)
    return psim

