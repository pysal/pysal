"""
Simulation of planar point processes

TODO

- inhibition process(es)
- optimize draws for complex windows
- documentation
"""

__author__ = "Serge Rey sjsrey@gmail.com"
__all__ = ['PointProcess', 'PoissonPointProcess', 'PoissonClusterPointProcess']

import numpy as np
import pysal.lib as ps
from numpy.random import poisson
from .pointpattern import PointPattern as PP


def runif_in_circle(n, radius=1.0, center=(0., 0.), burn=2, verbose=False):
    """
    Generate n points within a circle of given radius.

    Parameters
    ----------
    n             : int
                    Number of points.
    radius        : float
                    Radius of the circle.
    center        : tuple
                    Coordinates of the center.

    Returns
    -------
                  : array
                    (n+1, 2), coordinates of generated points as well as
                    the center.

    """

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
    """
    Point Process base class.

    Parameters
    ----------
    window        : :py:class:`~.window.Window`
                    Bounding geometric object to contain point process
                    realizations.
    n             : int
                    Size of each realization.
    samples       : list
                    Number of realizations.
    asPP          : bool
                    Control the data type of value in the "realizations"
                    dictionary. If True, the data type is point
                    pattern as defined in pointpattern.py; if False,
                    the data type is an two-dimensional array.

    Attributes
    ----------
    realizations  : dictionary
                    The key is the index of each realization, and the
                    value is simulated event points for each
                    realization. The data type of the value is
                    controlled by the parameter "asPP".
    parameters    : dictionary
                    Dictionary of a dictionary.
                    The key is the index of each realization, and the
                    value is a dictionary with the key 'n' and the
                    value size of each realization.

    """

    def __init__(self, window, n, samples, asPP=False, **args):

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

    def draw(self, parameter):
        """
        Generate a series of point coordinates within the given window.

        Parameters
        ----------
        parameter  : dictionary
                     Key: 'n'.
                     Value: size of the realization.

        Returns
        -------
                   : array
                     A series of point coordinates.

        """
        c = 0
        sample = []
        n = parameter['n']
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
    """
    Poisson point process including :math:`N`-conditioned CSR process and
    :math:`\lambda`-conditioned CSR process.

    Parameters
    ----------
    window        : :py:class:`~.window.Window`
                    Bounding geometric object to contain point process
                    realizations.
    n             : int
                    Size of each realization.
    samples       : list
                    Number of realizations.
    conditioning  : bool
                    If True, use the :math:`\lambda`-conditioned CSR process,
                    number of events would vary across realizations;
                    if False, use the :math:`N`-conditioned CSR process.
    asPP          : bool
                    Control the data type of value in the "realizations"
                    dictionary. If True, the data type is point
                    pattern as defined in pointpattern.py; if False,
                    the data type is an two-dimensional array.

    Attributes
    ----------
    realizations  : dictionary
                    The key is the index of each realization, and the
                    value is simulated event points for each
                    realization. The data type of the value is
                    controlled by the parameter "asPP".
    parameters    : dictionary
                    Dictionary of a dictionary.
                    The key is the index of each realization, and the
                    value is a dictionary with the key 'n' and the
                    value:
                    1. always equal to the parameter n in the case of
                    N-conditioned process.
                    For example, {0:{'n':100},1:{'n':100},2:{'n':100}}
                    2. randomly generated from a Possion process in
                    the case of lambda-conditioned process.
                    For example, {0:{'n':97},1:{'n':100},2:{'n':98}}

    Examples
    --------
    >>> import pysal.lib as ps
    >>> import numpy as np
    >>> from pointpats import Window
    >>> from pysal.lib.cg import shapely_ext

    Open the virginia polygon shapefile

    >>> va = ps.io.open(ps.examples.get_path("virginia.shp"))

    Create the exterior polygons for VA from the union of the county shapes

    >>> polys = [shp for shp in va]
    >>> state = shapely_ext.cascaded_union(polys)

    Create window from virginia state boundary

    >>> window = Window(state.parts)

    1. Simulate a :math:`N`-conditioned csr process in the same window (10
    points, 2 realizations)

    >>> np.random.seed(5)
    >>> samples1 = PoissonPointProcess(window, 10, 2, conditioning=False, asPP=False)
    >>> samples1.realizations[0] # the first realized event points
    array([[-81.80326547,  36.77687577],
           [-78.5166233 ,  37.34055832],
           [-77.21660795,  37.7491503 ],
           [-79.30361037,  37.40467853],
           [-78.61625258,  36.61234487],
           [-81.43369537,  37.13784646],
           [-80.91302108,  36.60834063],
           [-76.90806444,  37.95525903],
           [-76.33475868,  36.62635347],
           [-79.71621808,  37.27396618]])

    2. Simulate a :math:`\lambda`-conditioned csr process in the same window (10
    points, 2 realizations)

    >>> np.random.seed(5)
    >>> samples2 = PoissonPointProcess(window, 10, 2, conditioning=True, asPP=True)
    >>> samples2.realizations[0].n # the size of first realized point pattern
    10
    >>> samples2.realizations[1].n # the size of second realized point pattern
    13

    """

    def __init__(self, window, n, samples, conditioning=False, asPP=False):
        self.conditioning = conditioning
        super(PoissonPointProcess, self).__init__(window, n, samples, asPP)

    def setup(self):
        """
        Generate the number of events for each realization. If
        "conditioning" is False, all the event numbers are the same;
        if it is True, the event number is a random variable
        following a Poisson distribution.

        """

        self.parameters = {}
        if self.conditioning:
            lambdas = poisson(self.n, self.samples)
            for i, l in enumerate(lambdas):
                self.parameters[i] = {'n': l}
        else:
            for i in range(self.samples):
                self.parameters[i] = {'n': self.n}

    def realize(self, n):
        """
        Generate n points which are randomly and independently
        distributed in the minimum bounding box of "window".

        Parameters
        ----------
        n             : int
                        Number of point events.

        Returns
        -------
                      : array
                        (n,2), n point coordinates.

        """

        l, b, r, t = self.window.bbox
        xs = np.random.uniform(l, r, (n, 1))
        ys = np.random.uniform(b, t, (n, 1))
        return zip(xs, ys)


class PoissonClusterPointProcess(PointProcess):
    """
    Poisson cluster point process (Neyman Scott).
    Two stages: 
    1. parent CSR process: :math:`N`-conditioned or 
    :math:`\lambda`-conditioned. If parent events follow a 
    :math:`\lambda`-conditioned CSR process, 
    the number of parent events varies across realizations.
    2. child process: fixed number of points in circle centered 
    on each parent.

    Parameters
    ----------
    window        : :py:class:`~.window.Window`
                    Bounding geometric object to contain point process
                    realizations.
    n             : int
                    Size of each realization.
    parents       : int
                    Number of parents.
    radius        : float
                    Radius of the circle centered on each parent.
    samples       : list
                    Number of realizations.
    asPP          : bool
                    Control the data type of value in the "realizations"
                    dictionary. If True, the data type is point
                    pattern as defined in pointpattern.py; if False,
                    the data type is an two-dimensional array.
    conditioning  : bool
                    If True, use the :math:`lambda`-conditioned CSR process
                    for parent events, leading to varied number of
                    parent events across realizations;
                    if False, use the :math:`N`-conditioned CSR process.

    Attributes
    ----------
    children      : int
                    Number of childrens centered on each parent. Can
                    be considered as local intensity.
    num_parents   : dictionary
                    The key is the index of each realization. The
                    value is the number of parent events for each
                    realization.
    realizations  : dictionary
                    The key is the index of each realization, and the
                    value is simulated event points for each
                    realization. The data type of the value is
                    controlled by the parameter "asPP".
    parameters    : dictionary
                    Dictionary of a dictionary.
                    The key is the index of each realization, and the
                    value is a dictionary with the key 'n' and the
                    value always equal to the parameter n in the
                    case of
                    N-conditioned process.
                    For example, {0:{'n':100},1:{'n':100},2:{'n':100}}
                    2. randomly generated from a Possion process in
                    the case of lambda-conditioned process.
                    For example, {0:{'n':97},1:{'n':100},2:{'n':98}}

    Examples
    --------
    >>> import pysal.lib as ps
    >>> import numpy as np
    >>> from pointpats import Window
    >>> from pysal.lib.cg import shapely_ext

    Open the virginia polygon shapefile

    >>> va = ps.io.open(ps.examples.get_path("virginia.shp"))

    Create the exterior polygons for VA from the union of the county shapes

    >>> polys = [shp for shp in va]
    >>> state = shapely_ext.cascaded_union(polys)

    Create window from virginia state boundary

    >>> window = Window(state.parts)

    1. Simulate a Poisson cluster process of size 200 with 10 parents
    and 20 children within 0.5 units of each parent
    (parent events:  :math:`N`-conditioned CSR)
    
    >>> np.random.seed(10)
    >>> samples1 = PoissonClusterPointProcess(window, 200, 10, 0.5, 1, asPP=True, conditioning=False)
    >>> samples1.parameters # number of events for the realization
    {0: {'n': 200}}
    >>> samples1.num_parents #number of parent events for each realization
    {0: 10}
    >>> samples1.children # number of children events centered on each parent event
    20

    2. Simulate a Poisson cluster process of size 200 with 10 parents
    and 20 children within 0.5 units of each parent
    (parent events:  :math:`\lambda`-conditioned CSR)

    >>> np.random.seed(10)
    >>> samples2 = PoissonClusterPointProcess(window, 200, 10, 0.5, 1, asPP=True, conditioning=True)
    >>> samples2.parameters # number of events for the realization might not be equal to 200
    {0: {'n': 260}}
    >>> samples2.num_parents #number of parent events for each realization
    {0: 13}
    >>> samples2.children # number of children events centered on each parent event
    20

    """

    def __init__(self, window, n, parents, radius, samples, keep=False,
                 asPP=False, conditioning=False):

        self.conditioning = conditioning
        self.parents = parents
        self.children = int(np.ceil(n * 1. / parents))
        self.radius = radius
        self.keep = keep
        super(PoissonClusterPointProcess, self).__init__(window, n, samples,
                                                         asPP)

    def setup(self):
        """
        Generate the number of events for each realization. If
        "conditioning" is False, all the event numbers are the same;
        if it is True, the number of parents is a random variable
        following a Poisson distribution, resulting in varied number
        of events.

        """

        self.parameters = {}
        self.num_parents = {}
        if self.conditioning:
            lambdas = poisson(self.parents, self.samples)
            for i, l in enumerate(lambdas):
                num = l * self.children
                self.parameters[i] = {'n': num}
                self.num_parents[i] = l
        else:
            for i in range(self.samples):
                self.parameters[i] = {'n': self.n}
                self.num_parents[i] = self.parents

    def realize(self, n):
        """
        Generate n points which are distributed in a clustered
        fashion in the minimum bounding box of "window".

        Parameters
        ----------
        n             : int
                        Number of point events.

        Returns
        -------
        res           : array
                        (n,2), n point coordinates.

        """
        l, b, r, t = self.window.bbox
        d = self.radius
        # get parent points
        pxs = np.random.uniform(l, r, (int(n/self.children), 1))
        pys = np.random.uniform(b, t, (int(n/self.children), 1))
        cents = np.hstack((pxs, pys))
        # generate children points
        pnts = [runif_in_circle(self.children, d, center) for center in cents]
        res = np.vstack(np.asarray(pnts))
        if self.keep:
            res = np.vstack((np.asarray(cents), res))
        np.random.shuffle(res)  # so we don't truncate in a biased fashion
        return res
