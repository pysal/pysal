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


class Tau_Local:
    """
    Local version of the classic Tau.

    Decomposition of the classic Tau into local components.
    
    Parameters
    ----------
    x             : array
                    (n, ), first variable.
    y             : array
                    (n, ), second variable.

    Attributes
    ----------
    n             : int
                    number of observations.
    tau           : float
                    The classic Tau statistic.
    tau_local     : array
                    (n, ), local concordance (local version of the
                    classic tau).
    S             : array
                    (n ,n), concordance matrix, s_{i,j}=1 if observation i and j 
                    are concordant, s_{i,j}=-1 if observation i and j are discordant,
                    and s_{i,j}=0 otherwise.

    Notes
    -----
    The equation for calcualting local concordance statistic can be found in Rey
    (2016) [Rey2014]_ Equation (9). 


    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> f = ps.open(ps.examples.get_path("mexico.csv"))
    >>> vnames = ["pcgdp%d"%dec for dec in range(1940, 2010, 10)]
    >>> y = np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> r = y / y.mean(axis=0)
    >>> tau_local = Tau_Local(r[:,0],r[:,1])
    >>> tau_local.tau_local
    array([-0.03225806,  0.93548387,  0.80645161,  0.74193548,  0.93548387,
            0.74193548,  0.67741935,  0.41935484,  1.        ,  0.5483871 ,
            0.74193548,  0.93548387,  0.67741935,  0.74193548,  0.80645161,
            0.74193548,  0.5483871 ,  0.67741935,  0.74193548,  0.74193548,
            0.5483871 , -0.16129032,  0.93548387,  0.61290323,  0.67741935,
            0.48387097,  0.93548387,  0.61290323,  0.74193548,  0.41935484,
            0.61290323,  0.61290323])
    >>> tau_local.tau
    0.66129032258064513
    >>> tau_classic = Tau(r[:,0],r[:,1])
    >>> tau_classic.tau
    0.66129032258064513

    """

    def __init__(self, x, y):

        self.n = len(x)
        x = np.asarray(x)
        y = np.asarray(y)
        xx = x.reshape(self.n, 1)
        yy = y.reshape(self.n, 1)

        C = (xx - xx.T) * (yy - yy.T)
        self.S = -1 * (C < 0) + 1 * (C > 0)

        self.tau = self.S.sum()*1. / (self.n*(self.n-1))
        si = self.S.sum(axis=1)

        self.tau_local = si * 1. / (self.n - 1)


class Tau_Local_Neighbor:
    """
    Neighbor set LIMA.

    Local concordance relationships between a focal unit and its neighbors. A
    decomposition of local Tau into neighbor and non-neighbor components.
    
    Parameters
    ----------
    x              : array
                     (n, ), first variable.
    y              : array
                     (n, ), second variable.
    w              : W
                     spatial weights object.
    permutations   : int
                     number of random spatial permutations for computationally
                     based inference.

    Attributes
    ----------
    n              : int
                     number of observations.
    tau_local       : array
                     (n, ), local concordance (local version of the
                     classic tau).
    S              : array
                     (n ,n), concordance matrix, s_{i,j}=1 if observation i and j 
                     are concordant, s_{i,j}=-1 if observation i and j are discordant,
                     and s_{i,j}=0 otherwise.
    tau_ln         : array
                     (n, ), observed neighbor set LIMA values.
    tau_ln_sim     : array
                     (n, permutations), neighbor set LIMA values for permuted samples 
                     (if permutations>0).
    tau_ln_pvalues : array
                     (n, 1), pseudo p-values for observed neighbor set LIMA values
                     under the null that concordance relationship between the focal 
                     state and itsn eighbors is not different from what could be 
                     expected from randomly distributed rank changes.

    Notes
    -----
    The equation for calcualting neighbor set LIMA statistic can be found in Rey
    (2016) [Rey2014]_ Equation (16). 



    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> f = ps.open(ps.examples.get_path("mexico.csv"))
    >>> vnames = ["pcgdp%d"%dec for dec in range(1940, 2010, 10)]
    >>> y = np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> r = y / y.mean(axis=0)
    >>> regime = np.array(f.by_col['esquivel99'])
    >>> w = ps.weights.block_weights(regime)
    >>> res = Tau_Local_Neighbor(r[:,0],r[:,1],w,permutations=1000)
    >>> res.tau_ln
    array([-0.2       ,  1.        ,  1.        ,  1.        ,  0.33333333,
        0.6       ,  0.6       , -0.5       ,  1.        ,  1.        ,
        0.2       ,  0.33333333,  0.33333333,  0.5       ,  1.        ,
        1.        ,  1.        ,  0.        ,  0.6       , -0.33333333,
       -0.33333333, -0.6       ,  1.        ,  0.2       ,  0.        ,
        0.2       ,  1.        ,  0.6       ,  0.33333333,  0.5       ,
        0.5       , -0.2       ])
    >>> res.tau_ln_pvalues
    array([[ 0.1958042 ],
       [ 0.14885115],
       [ 0.33266733],
       [ 0.43056943],
       [ 0.000999  ],
       [ 0.12487512],
       [ 0.17882118],
       [ 0.002997  ],
       [ 0.000999  ],
       [ 0.25574426],
       [ 0.01098901],
       [ 0.000999  ],
       [ 0.05894106],
       [ 0.07692308],
       [ 0.0959041 ],
       [ 0.34465534],
       [ 0.44955045],
       [ 0.00899101],
       [ 0.1008991 ],
       [ 0.000999  ],
       [ 0.00799201],
       [ 0.05494505],
       [ 0.13386613],
       [ 0.03596404],
       [ 0.01498501],
       [ 0.0979021 ],
       [ 0.14485514],
       [ 0.24275724],
       [ 0.02297702],
       [ 0.33366633],
       [ 0.15584416],
       [ 0.000999  ]])

    """

    def __init__(self, x, y, w, permutations=0):
        
        x = np.asarray(x)
        y = np.asarray(y)
        res = Tau_Local(x, y)
        self.n = res.n
        self.S = res.S
        self.tau_local = res.tau_local   

        w.transform = 'b'
        W = w.full()[0]
        WS = W * self.S

        wsi = WS.sum(axis=1) 
        neighbor_num = np.array(w.cardinalities.values())
        self.tau_ln = wsi*1./neighbor_num  #Neighbor set LIMA

        if permutations > 0:
            tau_ln_sim = np.zeros((n,permutations))
            tau_ln_pvalues = np.zeros((n,1))
            for i in xrange(n):
                yr = np.zeros_like(y)
                xr = np.zeros_like(y)
                rids = range(self.n)
                rids.remove(i)
                larger = 0
                obs_i = self.tau_ln[i] #observed value of the ith neighbor set LIMA statistic

                for j in xrange(permutations):
                    pids = np.random.permutation(rids)
                    xr[i] = x[i]
                    xr[rids] = x[pids]
                    yr[i] = y[i]
                    yr[rids] = y[pids]
                    res = Tau_Local(xr,yr)
                    tau_ln_sim[i,j] = (W * res.S).sum(axis=1)[i]*1./neighbor_num[i]

                larger = (tau_ln_sim[i] >= obs_i).sum()
                if (permutations - larger) < larger:
                    larger = permutations - larger
                tau_ln_pvalues[i] = (larger+1.)/(1+permutations)
            self.tau_ln_sim = tau_ln_sim
            self.tau_ln_pvalues = tau_ln_pvalues




class Tau_Local_Neighborhood:
    """
    Neighborhood set LIMA.

    An extension of neighbor set LIMA. Consider local concordance relationships for 
    a subset of states, defined as the focal state and its neighbors.
    
    Parameters
    ----------
    x                  : array
                         (n, ), first variable.
    y                  : array
                         (n, ), second variable.
    w                  : W
                         spatial weights object.
    permutations       : int
                         number of random spatial permutations for computationally
                         based inference.

    Attributes
    ----------
    n                  : int
                         number of observations.
    tau_local          : array
                         (n, ), local concordance (local version of the
                         classic tau).
    S                  : array
                         (n ,n), concordance matrix, s_{i,j}=1 if observation i and j 
                         are concordant, s_{i,j}=-1 if observation i and j are discordant,
                         and s_{i,j}=0 otherwise.
    tau_lnhood         : array
                         (n, ), observed neighborhood set LIMA values.
    tau_lnhood_sim     : array
                         (n, permutations), neighborhood set LIMA values for permuted samples 
                         (if permutations>0).
    tau_lnhood_pvalues : array
                         (n, 1), pseudo p-values for observed neighborhood set LIMA values
                         under the null that the concordance relationships for a subset
                         of states, defined as the focal state and its neighbors, is 
                         different from what would be expected from randomly distributed 
                         rank changes.

    Notes
    -----
    The equation for calcualting neighborhood set LIMA statistic can be found in Rey
    (2016) [Rey2014]_ Equation (22). 

    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> f = ps.open(ps.examples.get_path("mexico.csv"))
    >>> vnames = ["pcgdp%d"%dec for dec in range(1940, 2010, 10)]
    >>> y = np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> r = y / y.mean(axis=0)
    >>> regime = np.array(f.by_col['esquivel99'])
    >>> w = ps.weights.block_weights(regime)
    >>> res = Tau_Local_Neighbor(r[:,0],r[:,1],w,permutations=1000)
    >>> res.tau_lnhood
    array([ 0.06666667,  0.6       ,  0.2       ,  0.8       ,  0.33333333,
        0.6       ,  0.6       ,  0.2       ,  1.        ,  0.06666667,
        0.06666667,  0.33333333,  0.33333333,  0.2       ,  1.        ,
        0.33333333,  0.33333333,  0.2       ,  0.6       ,  0.33333333,
        0.33333333,  0.06666667,  0.8       ,  0.06666667,  0.2       ,
        0.6       ,  0.8       ,  0.6       ,  0.33333333,  0.8       ,
        0.8       ,  0.06666667])
    >>> res.tau_lnhood_pvalues
    array([[ 0.05794206],
       [ 0.13486513],
       [ 0.02697303],
       [ 0.45754246],
       [ 0.02197802],
       [ 0.25074925],
       [ 0.29270729],
       [ 0.06193806],
       [ 0.000999  ],
       [ 0.01098901],
       [ 0.00799201],
       [ 0.01998002],
       [ 0.06093906],
       [ 0.03596404],
       [ 0.1028971 ],
       [ 0.05094905],
       [ 0.11088911],
       [ 0.04995005],
       [ 0.23676324],
       [ 0.02997003],
       [ 0.12087912],
       [ 0.06793207],
       [ 0.32867133],
       [ 0.01398601],
       [ 0.03396603],
       [ 0.35164835],
       [ 0.33966034],
       [ 0.28071928],
       [ 0.03396603],
       [ 0.37462537],
       [ 0.4965035 ],
       [ 0.01698302]])

    """

    def __init__(self, x, y, w, permutations=0):
        
        x = np.asarray(x)
        y = np.asarray(y)
        res = Tau_Local(x, y)
        self.n = res.n
        self.S = res.S
        self.tau_local = res.tau_local   

        w.transform = 'b'
        tau_lnhood = np.zeros(n)
        for i in xrange(self.n):
            neighbors_i = [i]
            neighbors_i.extend(w.neighbors[i])
            n_i = len(neighbors_i)
            sh_i = self.S[neighbors_i,:][:,neighbors_i]
            tau_lnhood[i] = sh_i.sum()*1./(n_i*(n_i-1)) #Neighborhood set LIMA
        self.tau_lnhood = tau_lnhood

        if permutations > 0:
            tau_lnhood_sim = np.zeros((n,permutations))
            tau_lnhood_pvalues = np.zeros((n,1))
            for i in xrange(n):
                obs_i = self.tau_lnhood[i]
                rids = range(n)
                rids.remove(i)
                larger = 0
                for j in xrange(permutations):
                    np.random.shuffle(rids)
                    neighbors_i = [i]
                    neighbors_i.extend(rids[:len(w.neighbors[i])])
                    n_i = len(neighbors_i)
                    sh_i = self.S[neighbors_i,:][:,neighbors_i]
                    tau_lnhood_sim[i,j] = sh_i.sum()*1./(n_i*(n_i-1))

                larger = (tau_lnhood_sim[i] >= obs_i).sum()
                if (permutations - larger) < larger:
                    larger = permutations - larger
                tau_lnhood_pvalues[i] = (larger+1.)/(1+permutations)

            self.tau_lnhood_sim = tau_lnhood_sim
            self.tau_lnhood_pvalues = tau_lnhood_pvalues


class Tau_Regional:
    """
    Inter and intraregional decomposition of the classic Tau.
    
    Parameters
    ---------- 
    x               : array
                      (n, ), first variable.
    y               : array
                      (n, ), second variable.
    regimes         : array
                      (n, ), ids of which regime an observation belongs to.
    permutations    : int
                      number of random spatial permutations for computationally
                      based inference.

    Attributes
    ----------
    n               : int
                      number of observations.
    S               : array
                      (n ,n), concordance matrix, s_{i,j}=1 if observation i and j 
                      are concordant, s_{i,j}=-1 if observation i and j are discordant,
                      and s_{i,j}=0 otherwise.
    tau_reg         : array
                      (k, k), observed concordance matrix with diagonal elements measuring 
                      concordance between units within a regime and the off-diagonal 
                      elements denoting concordance between observations from a specific 
                      pair of different regimes.
    tau_reg_sim     : array
                      (permutations, k, k), concordance matrices for permuted samples 
                      (if permutations>0).
    tau_reg_pvalues : array
                      (k, k), pseudo p-values for observed concordance matrix under the 
                      null that income mobility were random in its spatial distribution.

    Notes
    -----
    The equation for calcualting inter and intraregional Tau statistic can be 
    found in Rey (2016) [Rey2014]_ Equation (27). 

    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> f = ps.open(ps.examples.get_path("mexico.csv"))
    >>> vnames = ["pcgdp%d"%dec for dec in range(1940, 2010, 10)]
    >>> y = np.transpose(np.array([f.by_col[v] for v in vnames]))
    >>> r = y / y.mean(axis=0)
    >>> regime = np.array(f.by_col['esquivel99'])
    >>> res = Tau_Regional(income[:,0],income[:,-1],regime,permutations=1000)
    >>> res.tau_reg
    array([[ 1.        ,  0.25      ,  0.5       ,  0.6       ,  0.83333333,
         0.6       ,  1.        ],
       [ 0.25      ,  0.33333333,  0.5       ,  0.3       ,  0.91666667,
         0.4       ,  0.75      ],
       [ 0.5       ,  0.5       ,  0.6       ,  0.4       ,  0.38888889,
         0.53333333,  0.83333333],
       [ 0.6       ,  0.3       ,  0.4       ,  0.2       ,  0.4       ,
         0.28      ,  0.8       ],
       [ 0.83333333,  0.91666667,  0.38888889,  0.4       ,  0.6       ,
         0.73333333,  1.        ],
       [ 0.6       ,  0.4       ,  0.53333333,  0.28      ,  0.73333333,
         0.8       ,  0.8       ],
       [ 1.        ,  0.75      ,  0.83333333,  0.8       ,  1.        ,
         0.8       ,  0.33333333]])
    >>> res.tau_reg_pvalues
    array([[ 0.21878122,  0.06693307,  0.24975025,  0.34065934,  0.29370629,
         0.33266733,  0.2007992 ],
       [ 0.06693307,  0.12987013,  0.24475524,  0.06193806,  0.04795205,
         0.11888112,  0.32667333],
       [ 0.24975025,  0.24475524,  0.35564436,  0.0959041 ,  0.05694306,
         0.25874126,  0.12387612],
       [ 0.34065934,  0.06193806,  0.0959041 ,  0.05794206,  0.08791209,
         0.03896104,  0.21778222],
       [ 0.29370629,  0.04795205,  0.05694306,  0.08791209,  0.36363636,
         0.24975025,  0.004995  ],
       [ 0.33266733,  0.11888112,  0.25874126,  0.03896104,  0.24975025,
         0.37962038,  0.22677323],
       [ 0.2007992 ,  0.32667333,  0.12387612,  0.21778222,  0.004995  ,
         0.22677323,  0.12287712]])

    """

    def __init__(self, x, y, regime, permutations=0):


        
        x = np.asarray(x)
        y = np.asarray(y)
        res = Tau_Local(x, y)
        self.n = res.n
        self.S = res.S

        reg = np.array(regime).flatten()
        ur = np.unique(reg).tolist()
        k = len(ur)
        P = np.zeros((k,n))
        for i, r in enumerate(reg):
            P[ur.index(r),i] = 1 #construct P matrix

        w = pysal.weights.block_weights(regime)
        w.transform = 'b'
        W = w.full()[0]
        WH = np.ones((n,n)) - np.eye(n) - W

        self.tau_reg = _calc(W, WH, P, self.S) #inter and intraregional decomposition
        #of Tau for the observed value

        if permutations > 0:
            tau_reg_sim = np.zeros((permutations, k, k))
            larger = np.zeros((k,k))
            rids = np.arange(len(x))
            for i in xrange(permutations):
                np.random.shuffle(rids)
                res = Tau_Local(x[rids], y[rids])
                tau_reg_sim[i] = _calc(W, WH, P, res.S)
                larger += np.greater_equal(tau_reg_sim[i], self.tau_reg)
            m = np.less(permutations - larger, larger)
            pvalues = (1 + m * (permutations-larger) + (1-m) * larger) / (1. + permutations)
            self.tau_reg_sim = tau_reg_sim
            self.tau_reg_pvalues = pvalues

    def _calc(W, WH, P, S):

        # WS = W * S
        # PWSPt = np.dot(P,np.dot(WS,P.T)) 
        # WHS = WH * S
        # PWHSPt = np.dot(P,np.dot(WHS,P.T))
        # nomi = PWSPt + PWHSPt
        nomi = np.dot(P,np.dot(S,P.T))
        denomi = np.dot(P,np.dot(W,P.T)) + np.dot(P,np.dot(WH,P.T))
        T = nomi/denomi

        return T





