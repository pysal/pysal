"""
Distance statistics for planar point patterns

"""
__author__ = "Serge Rey sjsrey@gmail.com"
__all__ = ['DStatistic', 'G', 'F', 'J', 'K', 'L', 'Envelopes', 'Genv', 'Fenv', 'Jenv', 'Kenv', 'Lenv']

from .process import PoissonPointProcess as csr
import numpy as np
from matplotlib import pyplot as plt


class DStatistic(object):
    """
    Abstract Base Class for distance statistics.

    Parameters
    ----------
    name       : string
                 Name of the function. ("G", "F", "J", "K" or "L")

    Attributes
    ----------
    d          : array
                 The distance domain sequence.

    """
    def __init__(self, name):
        self.name = name

    def plot(self, qq=False):
        """
        Plot the distance function

        Parameters
        ----------
        qq: Boolean
            If False the statistic is plotted against distance. If Frue, the
            quantile-quantile plot is generated, observed vs. CSR.
        """

        # assuming mpl
        x = self.d
        if qq:
            plt.plot(self.ev, self._stat)
            plt.plot(self.ev, self.ev)
        else:
            plt.plot(x, self._stat, label='{}'.format(self.name))
            plt.ylabel("{}(d)".format(self.name))
            plt.xlabel('d')
            plt.plot(x, self.ev, label='CSR')
            plt.title("{} distance function".format(self.name))


class G(DStatistic):
    """
    Estimates the nearest neighbor distance distribution function G for a
    point pattern.

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    name       : string
                 Name of the function. ("G", "F", "J", "K" or "L")
    d          : array
                 The distance domain sequence.
    G          : array
                 The cumulative nearest neighbor distance distribution over d.

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
        self.ev = 1 - np.exp(-pp.lambda_window * np.pi * self.d * self.d)
        self.pp = pp
        super(G, self).__init__(name="G")


class F(DStatistic):
    """
    Estimates the empty space distribution function for a point pattern: F(d).

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    n          : int
                 Number of empty space points (random points).
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    d          : array
                 The distance domain sequence.
    G          : array
                 The cumulative empty space nearest event distance distribution
                 over d.

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
        self.ev = 1 - np.exp(-pp.lambda_window * np.pi * self.d * self.d)
        super(F, self).__init__(name="F")


class J(DStatistic):
    """
    Estimates the J function for a point pattern :cite:`VanLieshout1996`

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    n          : int
                 Number of empty space points (random points).
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    d          : array
                 The distance domain sequence.
    j          : array
                 F function over d.

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
        self.ev = self.j / self.j
        super(J, self).__init__(name="J")


class K(DStatistic):
    """
    Estimates the  K function for a point pattern.

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    d          : array
                 The distance domain sequence.
    j          : array
                 K function over d.

    """
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _k(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.k = self._stat = res[:, 1]
        self.ev = np.pi * self.d * self.d
        super(K, self).__init__(name="K")


class L(DStatistic):
    """
    Estimates the l function for a point pattern.

    Parameters
    ----------
    pp         : :class:`.PointPattern`
                 Point Pattern instance.
    intervals  : int
                 The length of distance domain sequence.
    dmin       : float
                 The minimum of the distance domain.
    dmax       : float
                 The maximum of the distance domain.
    d          : sequence
                 The distance domain sequence.
                 If d is specified, intervals, dmin and dmax are ignored.

    Attributes
    ----------
    d          : array
                 The distance domain sequence.
    l          : array
                 L function over d.
    """
    def __init__(self, pp, intervals=10, dmin=0.0, dmax=None, d=None):
        res = _l(pp, intervals, dmin, dmax, d)
        self.d = res[:, 0]
        self.l = self._stat = res[:, 1]
        super(L, self).__init__(name="L")

    def plot(self):
        # assuming mpl
        x = self.d
        plt.plot(x, self._stat, label='{}'.format(self.name))
        plt.ylabel("{}(d)".format(self.name))
        plt.xlabel('d')
        plt.title("{} distance function".format(self.name))


def _g(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    Estimate the nearest neighbor distances function G.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to maximum nearest neighor distance.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
             : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is the cumulative nearest neighbor distance distribution.

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
    F empty space function.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    n        : int
               Number of empty space points (random points).
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to maximum nearest neighor distance.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
             : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is corresponding F function.

    Notes
    -----
    See :class:`.F`

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
    J function: Ratio of hazard functions for F and G.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    n        : int
               Number of empty space points (random points).
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to maximum nearest neighor distance.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
             : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is corresponding J function.

    Notes
    -----
    See :class:`.J`

    """

    F = _f(pp, n, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    G = _g(pp, intervals=intervals, dmin=dmin, dmax=dmax, d=d)
    FC = 1 - F[:, 1]
    GC = 1 - G[:, 1]
    last_id = len(GC) + 1
    if np.any(FC == 0):
        last_id = np.where(FC == 0)[0][0]

    return np.vstack((F[:last_id, 0], GC[:last_id]/FC[:last_id])).T


def _k(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    Interevent K function.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    n        : int
               Number of empty space points (random points).
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to length of bounding box diagonal.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
    kcdf     : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is corresponding K function.

    Notes
    -----
    See :class:`.K`

    """

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


def _l(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    Interevent L function.

    Parameters
    ----------
    pp       : :class:`.PointPattern`
               Point Pattern instance.
    n        : int
               Number of empty space points (random points).
    intevals : int
               Number of intervals to evaluate F over.
    dmin     : float
               Lower limit of distance range.
    dmax     : float
               Upper limit of distance range. If dmax is None, dmax will be set
               to length of bounding box diagonal.
    d        : sequence
               The distance domain sequence. If d is specified, intervals, dmin
               and dmax are ignored.

    Returns
    -------
    kf       : array
               A 2-dimensional numpy array of 2 columns. The first column is
               the distance domain sequence for the point pattern. The second
               column is corresponding L function.

    Notes
    -----
    See :class:`.L`

    """

    kf = _k(pp, intervals, dmin, dmax, d)
    kf[:, 1] = np.sqrt(kf[:, 1] / np.pi) - kf[:, 0]
    return kf


class Envelopes(object):
    """
    Abstract base class for simulation envelopes.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  The maximum of the distance domain.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  1-alpha is the confidence level for the envelope.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is the specific function ("G", "F", "J",
                  "K" or "L") over the distance domain sequence for the
                  observed point pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    """
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

        # When calculating the J function for all the simulations, the length
        # of the returned interval domains might be different.

        if self.name == "J":
            res = []
            for p in reals:
                j = self.calc(reals[p])
                if j.shape[0] < self.d.shape[0]:
                    diff = self.d.shape[0]-j.shape[0]
                    for i in range(diff):
                        j = np.append(j, [[self.d[i+diff], np.inf]], axis=0)
                res.append(j)
            res = np.array(res)

        res = res[:, :, -1]
        res.sort(axis=0)
        nres = len(res)
        self.low = res[np.int(nres * self.pct/2.)]
        self.high = res[np.int(nres * (1-self.pct/2.))]
        self.mean = res.mean(axis=0)

    def calc(self, *args, **kwargs):
        print('implement in subclass')

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
    """
    Simulation envelope for G function.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  Upper limit of distance range. If dmax is None, dmax will be
                  set to maximum nearest neighbor distance.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  which means 95% confidence level for the envelopes.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is cumulative nearest neighbor distance
                  distribution (G function) for the observed point pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    Examples
    --------
    .. plot::

       >>> import pysal.lib as ps
       >>> from pointpats import Genv, PoissonPointProcess, Window
       >>> from pysal.lib.cg import shapely_ext
       >>> va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
       >>> polys = [shp for shp in va]
       >>> state = shapely_ext.cascaded_union(polys)
       >>> pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
       >>> csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
       >>> genv_bb = Genv(pp, realizations=csrs)
       >>> genv_bb.plot()

    """

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
    """
    Simulation envelope for F function.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    n           : int
                  Number of empty space points (random points).
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  Upper limit of distance range. If dmax is None, dmax will be
                  set to maximum nearest neighbor distance.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  which means 95% confidence level for the envelopes.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is F function for the observed point
                  pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    Examples
    --------
    .. plot::

       >>> import pysal.lib as ps
       >>> from pysal.lib.cg import shapely_ext
       >>> from pointpats import PoissonPointProcess,Window,Fenv
       >>> va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
       >>> polys = [shp for shp in va]
       >>> state = shapely_ext.cascaded_union(polys)
       >>> pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
       >>> csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
       >>> fenv = Fenv(pp, realizations=csrs)
       >>> fenv.plot()

    """
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
    """
    Simulation envelope for J function.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    n           : int
                  Number of empty space points (random points).
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  Upper limit of distance range. If dmax is None, dmax will be
                  set to maximum nearest neighbor distance.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  which means 95% confidence level for the envelopes.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is J function for the observed point
                  pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    Examples
    --------
    .. plot::

       >>> import pysal.lib as ps
       >>> from pointpats import Jenv, PoissonPointProcess, Window
       >>> from pysal.lib.cg import shapely_ext
       >>> va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
       >>> polys = [shp for shp in va]
       >>> state = shapely_ext.cascaded_union(polys)
       >>> pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
       >>> csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
       >>> jenv = Jenv(pp, realizations=csrs)
       >>> jenv.plot()

    """
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
    """
    Simulation envelope for K function.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  Upper limit of distance range. If dmax is None, dmax will be
                  set to maximum nearest neighbor distance.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  which means 95% confidence level for the envelope.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is K function for the observed point
                  pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    Examples
    --------
    .. plot::

       >>> import pysal.lib as ps
       >>> from pointpats import Kenv, PoissonPointProcess, Window
       >>> from pysal.lib.cg import shapely_ext
       >>> va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
       >>> polys = [shp for shp in va]
       >>> state = shapely_ext.cascaded_union(polys)
       >>> pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
       >>> csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
       >>> kenv = Kenv(pp, realizations=csrs)
       >>> kenv.plot()

    """
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
        return _k(pp, intervals=self.intervals, dmin=self.dmin, dmax=self.dmax,
                  d=self.d)


class Lenv(Envelopes):
    """
    Simulation envelope for L function.

    Parameters
    ----------
    pp          : :class:`.PointPattern`
                  Point Pattern instance.
    intervals   : int
                  The length of distance domain sequence. Default is 10.
    dmin        : float
                  The minimum of the distance domain.
    dmax        : float
                  Upper limit of distance range. If dmax is None, dmax will be
                  set to maximum nearest neighbor distance.
    d           : sequence
                  The distance domain sequence.
                  If d is specified, intervals, dmin and dmax are ignored.
    pct         : float
                  1-alpha, alpha is the significance level. Default is 0.05,
                  which means 95% confidence level for the envelopes.
    realizations: :class:`.PointProcess`
                  Point process instance with more than 1 realizations.

    Attributes
    ----------
    name        : string
                  Name of the function. ("G", "F", "J", "K" or "L")
    observed    : array
                  A 2-dimensional numpy array of 2 columns. The first column is
                  the distance domain sequence for the observed point pattern.
                  The second column is L function for the observed point
                  pattern.
    low         : array
                  A 1-dimensional numpy array. Lower bound of the simulation
                  envelope.
    high        : array
                  A 1-dimensional numpy array. Higher bound of the simulation
                  envelope.
    mean        : array
                  A 1-dimensional numpy array. Mean values of the simulation
                  envelope.

    Examples
    --------
    .. plot::

       >>> import pysal.lib as ps
       >>> from pointpats import Lenv, PoissonPointProcess, Window
       >>> from pysal.lib.cg import shapely_ext
       >>> va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
       >>> polys = [shp for shp in va]
       >>> state = shapely_ext.cascaded_union(polys)
       >>> pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
       >>> csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
       >>> lenv = Lenv(pp, realizations=csrs)
       >>> lenv.plot()

    """

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
        return _l(pp, intervals=self.intervals, dmin=self.dmin, dmax=self.dmax,
                  d=self.d)
