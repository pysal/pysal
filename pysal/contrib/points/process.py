import numpy as np
import pysal as ps
from numpy.random import poisson


class PointProcess(object):
    """docstring for PointProcess"""
    def __init__(self, window, n, samples, **args):
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

    def draw(self, parameters):
        c = 0
        realization = []
        n = parameters['n']
        while c < n:
            pnts = self.realize(n)
            pnts = [ps.cg.shapes.Point((x, y)) for x, y in pnts]
            pins = self.window.filter_contained(pnts)
            realization.extend(pins)
            c = len(realization)
        return np.array([np.asarray(p) for p in realization[:n]])

    def realize(self):
        pass

    def setup(self):
        pass


class PoissonPointProcess(PointProcess):
    """docstring for PoissonPointProcess"""
    def __init__(self, window, n, samples, conditioning=False):
        self.conditioning = conditioning
        super(PoissonPointProcess, self).__init__(window, n, samples)

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
