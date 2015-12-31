"""
Distance statistics for planar point patterns


TODO

- documentation
- testing

"""
__author__ = "Serge Rey sjsrey@gmail.com"

from process import PoissonPointProcess as csr
import numpy as np
from matplotlib import pyplot as plt


class DStatistic(object):
    """Abstract Base Class for distance statistics"""
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
    """Estimates the nearest neighbor distance distribution function for a
    point pattern: G(d)

    Parameters
    ----------

    pp: :py:class:`~.pointpattern.PointPattern`
        Point Pattern instance

    intervals: int
               The length of distance domain sequence

    dmin: float
          The minimum of the distance domain

    dmax: float
          The maximum of the distance domain

    d: sequence
       The distance domain sequence.
       If d is specified, dmin and dmax are ignored


    Attributes
    ----------
    d:  array
        The distance domain sequence

    G: array
        The cumulative nearest neighbor distance distribution over d


    Notes
    -----

    In the analysis of planar point processes, the estimate of :math:`G` is
    typically compared to the value expected from a completely spatial
    random (CSR) process given as:

    .. math::

            G(d) = 1 - e^{-\lambda \pi  d^2}

    where :math:`\lambda` is the intensity (points per unit area) of the point
    process and :math:`d` is distance.

    For a clustered pattern, the empirical function will be above the
    expectation, while for a uniform pattern the empirical function falls below
    the expectation.


    """

    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _g(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.G = self._stat = res[:, 1]
        super(G, self).__init__(name="G")


class F(DStatistic):
    """Estimates the empty space   distribution function for a point pattern: F(d)

    Parameters
    ----------

    pp: :py:class:`~.pointpattern.PointPattern`
        Point Pattern instance

    n: int
       number of empty space points

    intervals: int
               The length of distance domain sequence

    dmin: float
          The minimum of the distance domain

    dmax: float
          The maximum of the distance domain

    d: sequence
       The distance domain sequence.
       If d is specified, dmin and dmax are ignored


    Attributes
    ----------
    d:  array
        The distance domain sequence

    F: array
        The cumulative empty space nearest event distance distribution over d


    Notes
    -----

    In the analysis of planar point processes, the estimate of :math:`F` is
    typically compared to the value expected from a process that displays
    complete spatial randomness (CSR):

    .. math::

            F(d) = 1 - e^{-\lambda \pi  d^2}

    where :math:`\lambda` is the intensity (points per unit area) of the point
    process and :math:`d` is distance.

    The expectation is identical to the expectation for the :class:`G` function
    for a CSR process.  However, for a clustered pattern, the empirical G
    function will be below the expectation, while for a uniform pattern the
    empirical function falls above the expectation.

    """

    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _f(pp, n, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.F = self._stat = res[:, 1]
        super(F, self).__init__(name="F")


class J(DStatistic):
    """Estimates the  J function for a point pattern [VanLieshout1996]_

    Parameters
    ----------
    pp: :py:class:`~.pointpattern.PointPattern`
        Point Pattern instance

    n: int
       number of empty space points

    intevals: int
        number of intervals to evalue J over

    Returns
    -------
    j: array (intervals x 2)
         first column is d, second column is j(d)


    Notes
    -----

    The :math:`J` function is a ratio of the hazard functions defined for
    :math:`G` and :math:`F`:

    .. math::

            J(d) = \\frac{1-G(d) }{1-F(d)}

    where :math:`G(d)` is the nearest neighbor distance distribution function
    (see :class:`G`)
    and :math:`F(d)` is the empty space function (see :class:`F`).

    For a CSR process the J function equals 1. Empirical values larger than 1
    are indicative of uniformity, while values below 1 suggest clustering.


    """
    def __init__(self, pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _j(pp, n, intervals, dmin, dmax, d)
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


def _g(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    Estimate the nearest neighbor distances function


    Parameters
    ----------

    pp: PointPattern

    intervals: int
               The length of distance domain sequence

    dmin: float
          The minimum of the distance domain

    dmax: float
          The of the distance domain

    d: sequence
       The distance domain sequence.
       If d is specified, dmin and dmax are ignored


    Returns
    -------
    d:  array
        The distance domain sequence

    G: array
        The cumulative nearest neighbor distance distribution over d


    Notes
    -----
    See :class:`G`.

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


def _f(pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    F: empty space function

    Parameters
    ----------
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

    Notes
    -----
    See :class:`F`

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


def _j(pp, n=100, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    J: Ratio of hazard functions for F and G

    Parameters
    ----------
    pp: :py:class:`~.pointpattern.PointPattern`
        Point Pattern instance

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

    Notes
    -----
    See :class:`J`

    """

    F = _f(pp, n, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    G = _g(pp, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    FC = 1 - F[:, 1]
    GC = 1 - G[:, 1]
    last_id = len(GC) + 1
    if np.any(FC == 0):
        last_id = np.where(FC == 0)[0][0]

    return np.vstack((F[:last_id, 0], GC[:last_id]/FC[:last_id])).T


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
    """Abstrace base class for simulation envelopes"""
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
        print(res.shape)
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
        return _g(pp, intervals=self.intervals, dmin=self.dmin, dmax=self.dmax,
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
        return _f(pp, self.n, intervals=self.intervals, dmin=self.dmin,
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
        return _j(pp, self.n, intervals=self.intervals, dmin=self.dmin,
                  dmax=self.dmax, d=self.d)


class Kenv(Envelopes):
    """docstring for Kenv"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None,
                 pct=0.05, realizations=None):
        self.pp = pp
        self.intervals = intervals
        self.dmin = dmin
        self.dmax = dmax
        self.d = d
        self.pct = pct
        super(Kenv, self).__init__(pp, realizations=realizations, name="K")

    def calc(self, *args, **kwargs):
        pp = args[0]
        return k(pp, intervals=self.intervals, dmin=self.dmin,
                 dmax=self.dmax, d=self.d)


class Lenv(Envelopes):
    """docstring for Lenv"""
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None,
                 pct=0.05, realizations=None):
        self.pp = pp
        self.intervals = intervals
        self.dmin = dmin
        self.dmax = dmax
        self.d = d
        self.pct = pct
        super(Lenv, self).__init__(pp, realizations=realizations, name="L")

    def calc(self, *args, **kwargs):
        pp = args[0]
        return l(pp, intervals=self.intervals, dmin=self.dmin,
                 dmax=self.dmax, d=self.d)
