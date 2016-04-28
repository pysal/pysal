"""
Gamma index for spatial autocorrelation


"""
__author__ = "Luc Anselin <luc.anselin@asu.edu>"

import pysal
import numpy as np

__all__ = ['Gamma']

PERMUTATIONS = 999


class Gamma:
    """Gamma index for spatial autocorrelation


    Parameters
    ----------

    y               : array
                      variable measured across n spatial units
    w               : W
                      spatial weights instance
                      can be binary or row-standardized
    operation       : {'c', 's', 'a'}
                      attribute similarity function where,
                      'c' cross product
                      's' squared difference
                      'a' absolute difference
    standardize     : {'no', 'yes'}
                      standardize variables first
                      'no' keep as is
                      'yes' or 'y' standardize to mean zero and variance one
    permutations    : int
                      number of random permutations for calculation of pseudo-p_values

    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    op           : {'c', 's', 'a'}
                   attribute similarity function, as per parameters
                   attribute similarity function
    stand        : {'no', 'yes'}
                   standardization
    permutations : int
                   number of permutations
    gamma        : float
                   value of Gamma index
    sim_g        : array
                   (if permutations>0)
                   vector of Gamma index values for permuted samples
    p_sim_g      : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed Gamma is more extreme than under randomness
                   implemented as a two-sided test
    mean_g       : float
                   average of permuted Gamma values
    min_g        : float
                   minimum of permuted Gamma values
    max_g        : float
                   maximum of permuted Gamma values


    Examples
    --------

    use same example as for join counts to show similarity

    >>> import numpy as np
    >>> w=pysal.lat2W(4,4)
    >>> y=np.ones(16)
    >>> y[0:8]=0
    >>> np.random.seed(12345)
    >>> g = pysal.Gamma(y,w)
    >>> g.g
    20.0
    >>> round(g.g_z, 3)
    3.188
    >>> g.p_sim_g
    0.0030000000000000001
    >>> g.min_g
    0.0
    >>> g.max_g
    20.0
    >>> g.mean_g
    11.093093093093094
    >>> np.random.seed(12345)
    >>> g1 = pysal.Gamma(y,w,operation='s')
    >>> g1.g
    8.0
    >>> round(g1.g_z, 3)
    -3.706
    >>> g1.p_sim_g
    0.001
    >>> g1.min_g
    14.0
    >>> g1.max_g
    48.0
    >>> g1.mean_g
    25.623623623623622
    >>> np.random.seed(12345)
    >>> g2 = pysal.Gamma(y,w,operation='a')
    >>> g2.g
    8.0
    >>> round(g2.g_z, 3)
    -3.706
    >>> g2.p_sim_g
    0.001
    >>> g2.min_g
    14.0
    >>> g2.max_g
    48.0
    >>> g2.mean_g
    25.623623623623622
    >>> np.random.seed(12345)
    >>> g3 = pysal.Gamma(y,w,standardize='y')
    >>> g3.g
    32.0
    >>> round(g3.g_z, 3)
    3.706
    >>> g3.p_sim_g
    0.001
    >>> g3.min_g
    -48.0
    >>> g3.max_g
    20.0
    >>> g3.mean_g
    -3.2472472472472473
    >>> np.random.seed(12345)
    >>> def func(z,i,j):
    ...     q = z[i]*z[j]
    ...     return q
    ...
    >>> g4 = pysal.Gamma(y,w,operation=func)
    >>> g4.g
    20.0
    >>> round(g4.g_z, 3)
    3.188
    >>> g4.p_sim_g
    0.0030000000000000001

    """
    def __init__(self, y, w, operation='c', standardize='no', permutations=PERMUTATIONS):
        self.w = w
        self.y = y
        self.op = operation
        self.stand = standardize.lower()
        self.permutations = permutations
        if self.stand == 'yes' or self.stand == 'y':
            ym = np.mean(self.y)
            ysd = np.std(self.y)
            ys = (self.y - ym) / ysd
            self.y = ys
        self.g = self.__calc(self.y, self.op)

        if permutations:
            sim = [self.__calc(np.random.permutation(self.y), self.op)
                   for i in xrange(permutations)]
            self.sim_g = np.array(sim)
            self.min_g = np.min(self.sim_g)
            self.mean_g = np.mean(self.sim_g)
            self.max_g = np.max(self.sim_g)
            p_sim_g = self.__pseudop(self.sim_g, self.g)
            self.p_sim_g = p_sim_g
            self.g_z = (self.g - self.mean_g) / np.std(self.sim_g)

    def __calc(self, z, op):
        if op == 'c':     # cross-product
            zl = pysal.lag_spatial(self.w, z)
            g = (z * zl).sum()
        elif op == 's':   # squared difference
            zs = np.zeros(z.shape)
            z2 = z ** 2
            for i, i0 in enumerate(self.w.id_order):
                neighbors = self.w.neighbor_offsets[i0]
                wijs = self.w.weights[i0]
                zw = zip(neighbors, wijs)
                zs[i] = sum([wij * (z2[i] - 2.0 * z[i] * z[
                    j] + z2[j]) for j, wij in zw])
            g = zs.sum()
        elif op == 'a':    # absolute difference
            zs = np.zeros(z.shape)
            for i, i0 in enumerate(self.w.id_order):
                neighbors = self.w.neighbor_offsets[i0]
                wijs = self.w.weights[i0]
                zw = zip(neighbors, wijs)
                zs[i] = sum([wij * abs(z[i] - z[j]) for j, wij in zw])
            g = zs.sum()
        else:              # any previously defined function op
            zs = np.zeros(z.shape)
            for i, i0 in enumerate(self.w.id_order):
                neighbors = self.w.neighbor_offsets[i0]
                wijs = self.w.weights[i0]
                zw = zip(neighbors, wijs)
                zs[i] = sum([wij * op(z, i, j) for j, wij in zw])
            g = zs.sum()
        return g

    def __pseudop(self, sim, g):
        above = sim >= g
        larger = above.sum()
        psim = (larger + 1.) / (self.permutations + 1.)
        if psim > 0.5:
            psim = (self.permutations - larger + 1.) / (self.permutations + 1.)
        return psim

