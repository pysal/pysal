"""
Emprical Bayesian smoother using non-parametric mixture models
to specify the prior distribution of risks

This module is a python translation of mixlag function
in CAMAN R package that is originally written by Peter Schlattmann.
"""

__author__ = "Myunghwa Hwang <mhwang4@gmail.com>, Luc Anselin <luc.anselin@asu.edu>, Serge Rey <srey@asu.edu"

import numpy as np
from scipy.stats import poisson
import math
__all__ = ['NP_Mixture_Smoother']


class NP_Mixture_Smoother(object):
    """Empirical Bayesian Rate Smoother Using Mixture Prior Distributions
    It goes through 1) defining an initial set of subpopulations,
    2) VEM algorithm to determine the number of major subpopulations,
    3) EM algorithm, 4) combining simialr subpopulations, and 5) estimating
    EB rates from a mixture of prior distributions from subpopulation
    models.

    Parameters
    ----------
    e           : array-like
                  event variable measured across n spatial units
    b           : array-like
                  population at risk variable measured across n spatial units
    k           : integer
                  a seed number to specify the number of subpopulations
    acc         : float
                  convergence criterion; VEM and EM loops stop
                  when the increase of log likelihood is less than acc
    numiter     : integer
                  the maximum number of iterations for VEM and EM loops
    limit       : float
                  a parameter to cotrol the limit for combing subpopulation
                  models

    Attributes
    ----------
    e           : array
                  same as e in parameters
    b           : array
                  same as b in parameters
    n           : integer
                  the number of observations
    w           : float
                  a global weight value, 1 devided by n
    k           : integer
                  the number of subpopulations
    acc         : float
                  same as acc in parameters
    numiter     : integer
                  same as numiter in parameters
    limit       : float
                  same as limit in parameters
    p           : array
                  (k, 1), the proportions of individual subpopulations
    t           : array
                  (k, 1), prior risks of individual subpopulations
    r           : array
                  (n, 1), estimated rate values
    category    : array
                  (n, 1), indices of subpopulations to which each observation belongs

    Examples
    --------

    importing numpy, and NP_Mixture_Smoother

    >>> import numpy as np
    >>> from esda.mixture_smoothing import NP_Mixture_Smoother

    creating an arrary including event values

    >>> e = np.array([10, 5, 12, 20])

    creating an array including population-at-risk values

    >>> b = np.array([100, 150, 80, 200])

    applying non-parametric mixture smoothing to e and b

    >>> mixture = NP_Mixture_Smoother(e,b)

    extracting the smoothed rates through the property r of the NP_Mixture_Smoother instance

    >>> mixture.r
    array([0.10982278, 0.03445531, 0.11018404, 0.11018604])

    Checking the subpopulations to which each observation belongs

    >>> mixture.category
    array([1, 0, 1, 1])

    computing an initial set of prior distributions for the subpopulations

    >>> mixture.getSeed()
    (array([0.5, 0.5]), array([0.03333333, 0.15      ]))

    applying the mixture algorithm

    >>> mixture.mixalg()
    {'accuracy': 1.0, 'k': 1, 'p': array([1.]), 'grid': array([11.27659574]), 'gradient': array([0.]), 'mix_den': array([0., 0., 0., 0.])}

    estimating empirical Bayesian smoothed rates

    >>> mixture.getRateEstimates()
    (array([0.0911574, 0.0911574, 0.0911574, 0.0911574]), array([1, 1, 1, 1]))

    """

    def __init__(self, e, b, k=50, acc=1.E-7, numiter=5000, limit=0.01):
        self.e = np.asarray(e).flatten()
        self.b = np.asarray(b).flatten()
        self.n = len(e)
        self.w = 1. / self.n
        self.k = k
        self.acc = acc
        self.numiter = numiter
        self.limit = limit
        r = self.mixalg()
        self.p = r['p']
        self.t = r['grid']
        self.r, self.category = self.getRateEstimates()

    def getSeed(self):
        self.raw_r = self.e * 1.0 / self.b
        r_max, r_min = self.raw_r.max(), self.raw_r.min()
        r_diff = r_max - r_min
        step = r_diff / (self.k - 1)
        grid = np.arange(r_min, r_max + step, step)
        p = np.ones(self.k) * 1. / self.k
        return p, grid

    def getMixedProb(self, grid):
        mix = np.zeros((self.n, self.k))
        for i in range(self.n):
            for j in range(self.k):
                mix[i, j] = poisson.pmf(self.e[i], self.b[i] * grid[j])
        return mix

    def getGradient(self, mix, p):
        mix_p = mix * p
        mix_den = mix_p.sum(axis=1)
        obs_id = mix_den > 1.E-13
        for i in range(self.k):
            mix_den_len = len(mix_den)
            if (mix_den > 1.E-13).sum() == mix_den_len:
                mix_p[:, i] = (1. / mix_den_len) * mix[:, i] / mix_den
        gradient = []
        for i in range(self.k):
            gradient.append(mix_p[:, i][obs_id].sum())
        return np.array(gradient), mix_den

    def getMaxGradient(self, gradient):
        grad_max = gradient.max()
        grad_max_inx = gradient.argmax()
        if grad_max <= 0:
            return (0, 1)
        return (grad_max, grad_max_inx)

    def getMinGradient(self, gradient, p):
        p_fil = p > 1.E-8
        grad_fil = gradient[p_fil]
        grad_min = grad_fil.min()
        grad_min_inx = np.where(p_fil)[0][grad_fil.argmin()]
        if grad_min >= 1.E+7:
            return (1.E+7, 1)
        return (grad_min, grad_min_inx)

    def getStepsize(self, mix_den, ht):
        mix_den_fil = np.fabs(mix_den) > 1.E-7
        a = ht[mix_den_fil] / mix_den[mix_den_fil]
        b = 1.0 + a
        b_fil = np.fabs(b) > 1.E-7
        w = self.w
        sl = w * ht[b_fil] / b[b_fil]
        s11 = sl.sum()
        s0 = (w * ht).sum()

        step, oldstep = 0., 0.
        for i in range(50):
            grad1, grad2 = 0., 0.
            for j in range(self.n):
                a = mix_den[j] + step * ht[j]
            if math.fabs(a) > 1.E-7:
                b = ht[j] / a
                grad1 = grad1 + w * b
                grad2 = grad2 - w * b * b
            if math.fabs(grad2) > 1.E-10:
                step = step - grad1 / grad2
            if oldstep > 1.0 and step > oldstep:
                step = 1.
                break
            if grad1 < 1.E-7:
                break
            oldstep = step
        if step > 1.0:
            return 1.0
        return step

    def vem(self, mix, p, grid):
        res = {}
        for it in range(self.numiter):
            grad, mix_den = self.getGradient(mix, p)
            grad_max, grad_max_inx = self.getMaxGradient(grad)
            grad_min, grad_min_inx = self.getMinGradient(grad, p)
            ht = (mix[:, grad_max_inx] - mix[:, grad_min_inx]
                  ) * p[grad_min_inx]
            st = self.getStepsize(mix_den, ht)
            xs = st * p[grad_min_inx]
            p[grad_min_inx] = p[grad_min_inx] - xs
            p[grad_max_inx] = p[grad_max_inx] + xs
            if (grad_max - 1.0) < self.acc or it == (self.numiter - 1):
                res = {'k': self.k, 'accuracy': grad_max - 1.0, 'p': p, 'grid': grid, 'gradient': grad, 'mix_den': mix_den}
                break
        return res

    def update(self, p, grid):
        p_inx = p > 1.E-3
        new_p = p[p_inx]
        new_grid = grid[p_inx]
        self.k = len(new_p)
        return new_p, new_grid

    def em(self, nstep, grid, p):
        l = self.k - 1
        w, n, e, b = self.w, self.n, self.e, self.b
        if self.k == 1:
            s11 = (w * b / np.ones(n)).sum()
            s12 = (w * e / np.ones(n)).sum()
            grid[l] = s11 / s12
            p[l] = 1.
            mix = self.getMixedProb(grid)
            grad, mix_den = self.getGradient(mix, p)
            grad_max, grad_max_inx = self.getMaxGradient(grad)
            return {'accuracy': math.fabs(grad_max - 1), 'k': self.k, 'p': p, 'grid': grid, 'gradient': grad, 'mix_den': mix_den}
        else:
            res = {}
            for counter in range(nstep):
                mix = self.getMixedProb(grid)
                grad, mix_den = self.getGradient(mix, p)
                p = p * grad
                su = p[:-1].sum()
                p[l] = 1. - su
                for j in range(self.k):
                    mix_den_fil = mix_den > 1.E-10
                    f_len = len(mix_den_fil)
                    s11 = (w * e[mix_den_fil] / np.ones(f_len) * mix[mix_den_fil, j] / mix_den[mix_den_fil]).sum()
                    s12 = (w * b[mix_den_fil] * (mix[mix_den_fil, j] / np.ones(f_len)) / mix_den[mix_den_fil]).sum()
                    if s12 > 1.E-12:
                        grid[j] = s11 / s12
                grad_max, grad_max_inx = self.getMaxGradient(grad)
                res = {'accuracy': math.fabs(grad_max - 1.), 'step': counter + 1, 'k': self.k, 'p': p, 'grid': grid, 'gradient': grad, 'mix_den': mix_den}
                if res['accuracy'] < self.acc and counter > 10:
                    break
        return res

    def getLikelihood(self, mix_den):
        mix_den_fil = mix_den > 0
        r = np.log(mix_den[mix_den_fil]).sum()
        return r

    def combine(self, res):
        p, grid, k = res['p'], res['grid'], self.k
        diff = np.fabs(grid[:-1] - grid[1:])
        bp_seeds = (diff >= self.limit).nonzero()[0] + 1
        if k - len(bp_seeds) > 1:
            bp = [0]
            if len(bp_seeds) == 1:
                bp.append(bp_seeds[0])
                bp.append(k - 1)
            else:
                if bp_seeds[1] - bp_seeds[0] > 1:
                    bp.append(bp_seeds[0])
                for i in range(1, len(bp_seeds)):
                    if bp_seeds[i] - bp_seeds[i - 1] > 1:
                        bp.append(a[i])
            new_grid, new_p = [], []
            for i in range(len(bp) - 1):
                new_grid.append(grid[bp[i]])
                new_p.append(p[bp[i]:bp[i + 1]].sum())
            self.k = new_k = len(new_p)
            new_grid, new_p = np.array(new_grid), np.array(new_p)
            mix = self.getMixedProb(new_grid)
            grad, mix_den = self.getGradient(mix, new_p)
            res = self.em(1, new_grid, new_p)
            if res is not None:
                res['likelihood'] = self.getLikelihood(mix_den)
        return res

    def mixalg(self):
        e, b, k, n = self.e, self.b, self.k, self.n
        p, grid = self.getSeed()
        mix = self.getMixedProb(grid)
        vem_res = self.vem(mix, p, grid)
        p, grid, k = vem_res['p'], vem_res['grid'], vem_res['k']
        n_p, n_g = self.update(p, grid)
        em_res = self.em(self.numiter, n_g, n_p)
        com_res = self.combine(em_res)
        return com_res

    def getRateEstimates(self):
        mix = self.getMixedProb(self.t)
        mix_p = mix * self.p
        denom = mix_p.sum(axis=1)
        categ = (mix_p / denom.reshape((self.n, 1))).argmax(axis=1)
        r = (self.t * mix_p).sum(axis=1) / denom
        return r, categ
