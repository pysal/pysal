from process import PoissonPointProcess as csr
import numpy as np
from matplotlib import pyplot as plt


class DStatistic(object):
    """docstring for DStatistic"""
    def __init__(self, name):
        self.name = name

    def plot(self):
        # assuming mpl
        x = self.d
        plt.plot(x, self._stat, label='{}'.format(self.name))
        plt.ylabel("{}(d)".format(self.name))
        plt.xlabel('d')
        plt.title("{} distance function".format(self.name))


class G(DStatistic):
    """docstring for G"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = g(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.G = self._stat = res[:, 1]
        super(G, self).__init__(name="G")


class F(DStatistic):
    """docstring for F"""
    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
        res = f(pp, n, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.F = self._stat = res[:, 1]
        super(F, self).__init__(name="F")


class J(DStatistic):
    """docstring for J"""
    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
        res = j(pp, n, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.j = self._stat = res[:, 1]
        super(J, self).__init__(name="J")


class K(DStatistic):
    """docstring for K"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = k(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.k = self._stat = res[:, 1]
        super(K, self).__init__(name="K")


class L(DStatistic):
    """docstring for L"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = l(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.l = self._stat = res[:, 1]
        super(L, self).__init__(name="L")


def g(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    G function
    """

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


def k(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    if d is None:
        # use length of bounding box diagonal as max distance
        bb = pp.mbb
        dbb = np.sqrt((bb[0]-bb[2])**2 + (bb[1]-bb[3])**2)
        w = dbb/intervals
        if dmax:
            w = dmax/intervals
    d = [w*i for i in range(intervals + 2)]
    den = pp.lambda_window * pp.n * 2.
    kcdf = np.asarray([(di, len(pp.tree.query_pairs(di))/den) for di in d])
    return kcdf


def l(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    kf = k(pp, intervals, dmin, dmax, d)
    kf[:, 1] = np.sqrt(kf[:, 1] / np.pi) - kf[:, 0]
    return kf


class Envelopes(object):
    """docstring for Envelopes"""
    def __init__(self, *args,  **kwargs):
        # setup arguments
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
        plt.plot(x, self.observed[:, 1], label='{}'.format(self.name))
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
    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None,
                 pct=0.05, realizations=None):
        self.pp = pp
        self.n = n
        self.intervals = intervals
        self.dmin = dmin
        self.dmax = dmax
        self.d = d
        self.pct = pct
        super(Fenv, self).__init__(pp, realizations=realizations, name="F")

    def calc(self, *args, **kwargs):
        pp = args[0]
        return f(pp, self.n, intervals=self.intervals, dmin=self.dmin,
                 dmax=self.dmax, d=self.d)


class Jenv(Envelopes):
    """docstring for Jenv"""
    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None,
                 pct=0.05, realizations=None):
        self.pp = pp
        self.n = n
        self.intervals = intervals
        self.dmin = dmin
        self.dmax = dmax
        self.d = d
        self.pct = pct
        super(Jenv, self).__init__(pp, realizations=realizations, name="J")

    def calc(self, *args, **kwargs):
        pp = args[0]
        return j(pp, self.n, intervals=self.intervals, dmin=self.dmin,
                 dmax=self.dmax, d=self.d)


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
