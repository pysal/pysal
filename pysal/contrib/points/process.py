import numpy as np
import pysal as ps


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
        for sample in range(samples):
            self.realizations[sample] = self.draw()

    def draw(self):
        c = 0
        realization = []
        while c < self.n:
            pnts = self.realize()
            pnts = [ps.cg.shapes.Point((x, y)) for x, y in pnts]
            pins = self.window.filter_contained(pnts)
            # pins = [pnt for pnt in pnts if self.contains_point(pnt)]
            realization.extend(pins)
            c = len(realization)
        return np.array([np.asarray(p) for p in realization[:self.n]])

    def realize(self):
        pass


class PoissonPointProcess(PointProcess):
    """docstring for PoissonPointProcess"""
    def __init__(self, window, n, samples,  conditioning=False):
        self.conditioning = conditioning
        if conditioning:
            print('conditioning')
        super(PoissonPointProcess, self).__init__(window, n, samples)

    def realize(self):
        l, b, r, t = self.window.bbox
        xs = np.random.uniform(l, r, (self.n, 1))
        ys = np.random.uniform(b, t, (self.n, 1))
        return zip(xs, ys)
