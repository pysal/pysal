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


class PoissonClusterPointProcess(PointProcess):
    """docstring for PoissonPointProcess"""
    def __init__(self, window, n, parents, radius, samples, keep=False):
        self.parents = parents
        self.children = np.ceil(n * 1. / parents)
        self.radius = radius
        self.keep = keep
        super(PoissonClusterPointProcess, self).__init__(window, n, samples)

    def setup(self):
        parameters = {}
        for i in range(self.samples):
            parameters[i] = {'n': self.n}
        self.parameters = parameters

    def realize(self, n):
        l, b, r, t = self.window.bbox
        d = self.radius
        pxs = np.random.uniform(l, r, (self.parents, 1))
        pys = np.random.uniform(b, t, (self.parents, 1))
        cxs = [np.random.uniform(px-d, px+d, (self.children, 1)) for px in pxs]
        cys = [np.random.uniform(py-d, py+d, (self.children, 1)) for py in pys]
        # Need to filter for ensuring children are within d units of parent
        n_points = self.children * self.parents
        xs = []
        ys = []
        if self.keep:
            n_points += self.parents
            xs.extend(pxs)
            ys.extend(pys)
        for p in range(self.parents):
            xs.extend(cxs[p])
            ys.extend(cys[p])
        res = np.hstack((np.asarray(xs), np.asarray(ys)))
        np.random.shuffle(res)
        return res
