"""
Geary's C statistic for spatial autocorrelation
"""
__author__ = "Sergio J. Rey <srey@asu.edu> "

import numpy as np
import scipy.stats as stats

__all__ = ['Geary']


class Geary:
    """
    Global Geary C Autocorrelation statistic

    Parameters
    ----------
    y              : array
                     (n, 1) attribute vector
    w              : W
                     spatial weights
    transformation : {'B', 'R', 'D', 'U', 'V'}
                     weights transformation, default is binary.
                     Other options include "R": row-standardized, "D":
                     doubly-standardized, "U": untransformed (general
                     weights), "V": variance-stabilizing.
    permutations   : int
                     number of random permutations for calculation of
                     pseudo-p_values

    Attributes
    ----------
    y              : array
                     original variable
    w              : W
                     spatial weights
    permutations   : int
                     number of permutations
    C              : float
                     value of statistic
    EC             : float
                     expected value
    VC             : float
                     variance of G under normality assumption
    z_norm         : float
                     z-statistic for C under normality assumption
    z_rand         : float
                     z-statistic for C under randomization assumption
    p_norm         : float
                     p-value under normality assumption (one-tailed)
    p_rand         : float
                     p-value under randomization assumption (one-tailed)
    sim            : array
                     (if permutations!=0)
                     vector of I values for permutated samples
    p_sim          : float
                     (if permutations!=0)
                     p-value based on permutations (one-tailed)
                     null: sptial randomness
                     alternative: the observed C is extreme
                     it is either extremely high or extremely low
    EC_sim         : float
                     (if permutations!=0)
                     average value of C from permutations
    VC_sim         : float
                     (if permutations!=0)
                     variance of C from permutations
    seC_sim        : float
                     (if permutations!=0)
                     standard deviation of C under permutations.
    z_sim          : float
                     (if permutations!=0)
                     standardized C based on permutations
    p_z_sim        : float
                     (if permutations!=0)
                     p-value based on standard normal approximation from
                     permutations (one-tailed)

    Examples
    --------
    >>> import pysal
    >>> w = pysal.open(pysal.examples.get_path("book.gal")).read()
    >>> f = pysal.open(pysal.examples.get_path("book.txt"))
    >>> y = np.array(f.by_col['y'])
    >>> c = Geary(y,w,permutations=0)
    >>> print round(c.C,7)
    0.3330108
    >>> print round(c.p_norm,7)
    9.2e-05
    >>>
    """
    def __init__(self, y, w, transformation="r", permutations=999):
        self.n = len(y)
        self.y = y
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.__moments()
        xn = xrange(len(y))
        self.xn = xn
        self.y2 = y * y
        yd = y - y.mean()
        yss = sum(yd * yd)
        self.den = yss * self.w.s0 * 2.0
        self.C = self.__calc(y)
        de = self.C - 1.0
        self.EC = 1.0
        self.z_norm = de / self.seC_norm
        self.z_rand = de / self.seC_rand
        if de > 0:
            self.p_norm = 1 - stats.norm.cdf(self.z_norm)
            self.p_rand = 1 - stats.norm.cdf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)


        if permutations:
            sim = [self.__calc(np.random.permutation(self.y))
                   for i in xrange(permutations)]
            self.sim = sim = np.array(sim)
            above = sim >= self.C
            larger = sum(above)
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.EC_sim = sum(sim) / permutations
            self.seC_sim = np.array(sim).std()
            self.VC_sim = self.seC_sim ** 2
            self.z_sim = (self.C - self.EC_sim) / self.seC_sim
            self.p_z_sim = 1 - stats.norm.cdf(np.abs(self.z_sim))

    def __moments(self):
        y = self.y
        n = self.n
        w = self.w
        s0 = w.s0
        s1 = w.s1
        s2 = w.s2
        s02 = s0 * s0

        yd = y - y.mean()
        k = (1 / (sum(yd ** 4)) * ((sum(yd ** 2)) ** 2))
        vc_rand = (1 / (n * ((n - 2) ** 2) * s02)) * \
            ((((n - 1) * s1) * (n * n - 3 * n + 3 - (n - 1) * k))
             - ((.25 * (n - 1) * s2) * (n * n + 3 * n - 6 -
                (n * n - n + 2) * k))
                + (s02 * (n * n - 3 - ((n - 1) ** 2) * k)))
        vc_norm = ((1 / (2 * (n + 1) * s02)) *
                   ((2 * s1 + s2) * (n - 1) - 4 * s02))

        self.VC_rand = vc_rand
        self.VC_norm = vc_norm
        self.seC_rand = vc_rand ** (0.5)
        self.seC_norm = vc_norm ** (0.5)

    def __calc(self, y):
        ys = np.zeros(y.shape)
        y2 = y ** 2
        for i, i0 in enumerate(self.w.id_order):
            neighbors = self.w.neighbor_offsets[i0]
            wijs = self.w.weights[i0]
            z = zip(neighbors, wijs)
            ys[i] = sum([wij * (y2[i] - 2 * y[i] * y[j] + y2[j])
                         for j, wij in z])
        a = (self.n - 1) * sum(ys)
        return a / self.den


