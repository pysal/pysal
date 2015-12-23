from process import PoissonPointProcess as csr
import numpy as np
from matplotlib import pyplot as plt


def g(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    G function
    """

    # res = pp.G(intervals, dmin, dmax, d)
    if d is None:
        w = pp.max_nnd/intervals
        if dmax:
            w = dmax/intervals
        d = [w*i for i in range(intervals + 2)]
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [nndi for nndi in pp.nnd if nndi <= d_i]
        cdf[i] = len(smaller)*1./pp.n
    return np.vstack((d, cdf)).T


def f(pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    F: empty space function

    Arguments
    ---------
    n: int
       number of empty space points
    intevals: int
        number of intervals to evalue F over
    dmin: float
           lower limit of distance range
    dmax: float
           upper limit of distance range
           if dmax is None dmax will be set to maxnnd
    d:   array-like
         domain for F function

    Returns
    -------
    cdf: array (intervals x 2)
         first column is d, second column is cdf(d)

    """

    # get a csr pattern in window of pp
    c = csr(pp.window, n, 1, asPP=True).realizations[0]
    # for each point in csr pattern find the closest point in pp and the
    # associated distance
    nnids, nnds = pp.knn_other(c, k=1)

    if d is None:
        w = pp.max_nnd/intervals
        if dmax:
            w = dmax/intervals
        d = [w*i for i in range(intervals + 2)]
    cdf = [0] * len(d)

    for i, d_i in enumerate(d):
        smaller = [nndi for nndi in nnds if nndi <= d_i]
        cdf[i] = len(smaller)*1./n
    return np.vstack((d, cdf)).T


def j(pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    J: scaled G function

    Arguments
    ---------
    n: int
       number of empty space points
    intevals: int
        number of intervals to evalue F over

    Returns
    -------
    cdf: array (intervals x 2)
         first column is d, second column is cdf(d)
    """
    F = f(pp, n, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    G = g(pp, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    FC = 1 - F[:, 1]
    GC = 1 - G[:, 1]
    last_id = len(GC) + 1
    if np.any(FC == 0):
        last_id = np.where(FC == 0)[0][0]

    return np.vstack((F[:last_id, 0], FC[:last_id]/GC[:last_id])).T


class Envelopes(object):
    """docstring for Envelopes"""
    def __init__(self, *args,  **kwargs):
        print(args)
        print(kwargs)
        # setup arguments
        self.args = args
        self.kwargs = kwargs
        self.name = kwargs['name']

        # calculate observed function
        self.pp = args[0]
        self.observed = self.calc(*args, **kwargs)
        self.d = self.observed[:, 0]  # domain to be used in all realizations

        # do realizations
        self.mapper(kwargs['realizations'])

    def mapper(self, realizations):
        reals = realizations.realizations
        res = np.asarray([self.calc(reals[p]) for p in reals])
        res = res[:, :, -1]
        res.sort(axis=0)
        nres = len(res)
        self.low = res[np.int(nres * self.pct)]
        self.high = res[np.int(nres * (1-self.pct))]
        self.mean = res.mean(axis=0)

    def calc(self, *args, **kwargs):
        print('implment in subclass')

    def plot(self):
        # assuming mpl
        x = self.d
        plt.plot(x, self.observed[:, 1], label='G')
        plt.plot(x, self.mean, 'g-.', label='CSR')
        plt.plot(x, self.low, 'r-.', label='LB')
        plt.plot(x, self.high, 'r-.', label="UB")
        plt.ylabel("{}(d)".format(self.name))
        plt.xlabel('d')
        plt.title("{} Simulation Envelopes".format(self.name))
        plt.legend(loc=0)


class Genv(Envelopes):
    """docstring for Genv"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None, pct=0.05,
                 realizations=None):
        self.pp = pp
        self.intervals = intervals
        self.dmin = dmin
        self.dmax = dmax
        self.d = d
        self.pct = pct
        super(Genv, self).__init__(pp, realizations=realizations, name="G")

    def calc(self, *args, **kwargs):
        pp = args[0]
        return g(pp, intervals=self.intervals, dmin=self.dmin, dmax=self.dmax,
                 d=self.d)


class Fenv(Envelopes):
    """docstring for Fenv"""
    def __init__(self, arg, *args, **kwargs):
        print("Fenv, arg=", arg)
        super(Fenv, self).__init__(*args, **kwargs)

    def calc(self):
        return 10


def g_envelopes(pp, intervals=10, d=None, reps=99, pct=0.05):
    obs = g(pp, intervals=intervals, d=d)
    sim = csr(pp.window, pp.n, reps, asPP=True)
    gs = np.asarray([g(p, d=obs[:, 0]) for p in sim.realizations.values()])
    gs = gs[:, :, -1]
    gs.sort(axis=0)
    low = gs[np.int(reps * pct)]
    high = gs[np.int(reps * (1-pct))]
    x = obs[:, 0]
    gobs = obs[:, 1]
    mean = gs.mean(axis=0)

    return [gs, x, gobs, mean, low, high]


def f_envelopes(pp, intervals=10, d=None, reps=99, pct=0.05):
    obs = f(pp, intervals=intervals, d=d)
    sim = csr(pp.window, pp.n, reps, asPP=True)
    fs = np.asarray([f(p, d=obs[:, 0]) for p in sim.realizations.values()])
    fs = fs[:, :, -1]
    fs.sort(axis=0)
    low = fs[np.int(reps * pct)]
    high = fs[np.int(reps * (1-pct))]
    x = obs[:, 0]
    fobs = obs[:, 1]
    mean = fs.mean(axis=0)

    return [fs, x, fobs, mean, low, high]


def j_envelopes(pp, n=100, intervals=10, d=None, reps=99, pct=0.05):
    obs = j(pp, n, intervals=intervals, d=d)
    sim = csr(pp.window, pp.n, reps, asPP=True)
    js = np.asarray([j(p, n, d=obs[:, 0]) for p in sim.realizations.values()])
    js = js[:, :, -1]
    js.sort(axis=0)
    low = js[np.int(reps * pct)]
    high = js[np.int(reps * (1-pct))]
    x = obs[:, 0]
    gobs = obs[:, 1]
    mean = js.mean(axis=0)

    return [js, x, gobs, mean, low, high]
