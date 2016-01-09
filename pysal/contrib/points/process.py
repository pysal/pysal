"""
Simulation of planar point processes

TODO

- inhibition process(es)
- optimize draws for complex windows
- documentation

- $\lambda$-conditioned CSR for parent events?
"""

__author__ = "Serge Rey sjsrey@gmail.com"

import numpy as np
import pysal as ps
from numpy.random import poisson
from pointpattern import PointPattern as PP


def runif_in_circle(n, radius=1.0, center=(0., 0.), burn=2, verbose=False):
    good = np.zeros((n, 2), float)
    c = 0
    r = radius
    r2 = r * r
    it = 0
    while c < n:
        x = np.random.uniform(-r, r, (burn*n, 1))
        y = np.random.uniform(-r, r, (burn*n, 1))
        ids = np.where(x*x + y*y <= r2)
        candidates = np.hstack((x, y))[ids[0]]
        nc = candidates.shape[0]
        need = n - c
        if nc > need:  # more than we need
            good[c:] = candidates[:need]
        else:  # use them all and keep going
            good[c:c+nc] = candidates
        c += nc
        it += 1
    if verbose:
        print('Iterations: {}'.format(it))
    return good + np.asarray(center)


class PointProcess(object):
    """docstring for PointProcess"""
    def __init__(self, window, n, samples, asPP=False, **args):
        """
        Parameters
        ---------

        window: window to contain point process realizations

        n: int size of each realization

        samples: number of realizations
        """
        super(PointProcess, self).__init__()
        self.window = window
        self.n = n
        self.samples = samples
        self.args = args
        self.realizations = {}
        self.setup()
        for sample in range(samples):
            self.realizations[sample] = self.draw(self.parameters[sample])
        if asPP:
            for sample in self.realizations:
                points = self.realizations[sample]
                self.realizations[sample] = PP(points, window=self.window)

    def draw(self, parameters):
        c = 0
        sample = []
        n = parameters['n']
        while c < n:
            pnts = self.realize(n)
            pnts = [ps.cg.shapes.Point((x, y)) for x, y in pnts]
            pins = self.window.filter_contained(pnts)
            sample.extend(pins)
            c = len(sample)
        return np.array([np.asarray(p) for p in sample[:n]])

    def realize(self):
        pass

    def setup(self):
        pass


class PoissonPointProcess(PointProcess):
    """docstring for PoissonPointProcess"""
    def __init__(self, window, n, samples, conditioning=False, asPP=False):
        self.conditioning = conditioning
        super(PoissonPointProcess, self).__init__(window, n, samples, asPP)

    def setup(self):
        self.parameters = {}
        if self.conditioning:
            lambdas = poisson(self.n, self.samples)
            for i, l in enumerate(lambdas):
                self.parameters[i] = {'n': l}
        else:
            for i in range(self.samples):
                self.parameters[i] = {'n': self.n}

    def realize(self, n):
        l, b, r, t = self.window.bbox
        xs = np.random.uniform(l, r, (n, 1))
        ys = np.random.uniform(b, t, (n, 1))
        return zip(xs, ys)


class PoissonClusterPointProcess(PointProcess):
    """docstring for PoissonPointProcess"""
    def __init__(self, window, n, parents, radius, samples, keep=False,
                 asPP=False):
        self.parents = parents
        self.children = np.ceil(n * 1. / parents)
        self.radius = radius
        self.keep = keep
        super(PoissonClusterPointProcess, self).__init__(window, n, samples,
                                                         asPP)

    def setup(self):
        parameters = {}
        for i in range(self.samples):
            parameters[i] = {'n': self.n}
        self.parameters = parameters

    def realize(self, n):
        l, b, r, t = self.window.bbox
        d = self.radius
        # get parent points
        pxs = np.random.uniform(l, r, (self.parents, 1))
        pys = np.random.uniform(b, t, (self.parents, 1))
        cents = np.hstack((pxs, pys))
        # generate children points
        pnts = [runif_in_circle(self.children, d, center) for center in cents]
        res = np.vstack(np.asarray(pnts))
        if self.keep:
            res = np.vstack((np.asarray(cents), res))
        np.random.shuffle(res)  # so we don't truncate in a biased fashion
        return res
