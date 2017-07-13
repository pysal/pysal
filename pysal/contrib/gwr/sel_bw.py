# GWR Bandwidth selection class

#x_glob parameter does not yet do anything; it is for semiparametric


__author__ = "Taylor Oshan Tayoshan@gmail.com"

from kernels import *
from search import golden_section, equal_interval
from gwr import GWR
from pysal.contrib.glm.family import Gaussian, Poisson, Binomial
import pysal.spreg.user_output as USER
from diagnostics import get_AICc, get_AIC, get_BIC, get_CV
from scipy.spatial.distance import pdist, squareform
from pysal.common import KDTree
import numpy as np

kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6:adapt_exp}
getDiag = {'AICc': get_AICc,'AIC':get_AIC, 'BIC': get_BIC, 'CV': get_CV}

class Sel_BW(object):
    """
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset         : array
                     n*1, offset variable for Poisson model
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.


    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel         : string
                     type of kernel used and wether fixed or adaptive
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search         : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval increment used in interval search
    tol            : float
                     tolerance used to determine convergence
    max_iter       : integer
                     max interations if no convergence to tol
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    Examples
    ________

    >>> import pysal
    >>> from pysal.contrib.gwr.sel_bw import Sel_BW
    >>> data = pysal.open(pysal.examples.get_path('GData_utm.csv'))
    >>> coords = zip(data.bycol('X'), data.by_col('Y')) 
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    
    #Golden section search AICc - adaptive bisquare
    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print bw
    93.0

    #Golden section search AIC - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
    >>> print bw
    50.0

    #Golden section search BIC - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
    >>> print bw
    62.0

    #Golden section search CV - adaptive Gaussian
    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
    >>> print bw
    68.0

    #Interval AICc - fixed bisquare
    >>>  sel = Sel_BW(coords, y, X, fixed=True).
    >>>  bw = sel.search(search='interval', bw_min=211001.0, bw_max=211035.0, interval=2) 
    >>> print bw
    211025.0

    """
    def __init__(self, coords, y, X_loc, X_glob=None, family=Gaussian(),
            offset=None, kernel='bisquare', fixed=False, constant=True):
        self.coords = coords
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        if offset is None:
            self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        self.constant = constant

    def search(self, search='golden_section', criterion='AICc', bw_min=0.0,
            bw_max=0.0, interval=0.0, tol=1.0e-6, max_iter=200):
        """
        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search         : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol

        Returns
        -------
        bw             : scalar or array
                         optimal bandwidth value
        """
        self.search = search
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter

        if self.fixed:
            if self.kernel == 'gaussian':
                ktype = 1
            elif self.kernel == 'bisquare':
                ktype = 3
            elif self.kernel == 'exponential':
                ktype = 5
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)
        else:
            if self.kernel == 'gaussian':
                ktype = 2
            elif self.kernel == 'bisquare':
                ktype = 4
            elif self.kernel == 'exponential':
                ktype = 6
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)

        function = lambda bw: getDiag[criterion](
                GWR(self.coords, self.y, self.x_loc, bw, family=self.family,
                    kernel=self.kernel, fixed=self.fixed, offset=self.offset).fit())

        if ktype % 2 == 0:
            int_score = True
        else:
            int_score = False
        self.int_score = int_score

        self._bw()

        return self.bw[0]

    def _bw(self):
        gwr_func = lambda bw: getDiag[self.criterion](
                GWR(self.coords, self.y, self.X_loc, bw, family=self.family,
                    kernel=self.kernel, fixed=self.fixed, constant=self.constant).fit())
        if self.search == 'golden_section':
            a,c = self._init_section(self.X_glob, self.X_loc, self.coords,
                    self.constant)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                    self.max_iter, self.int_score)
        elif self.search == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                    gwr_func, self.int_score)
        else:
            raise TypeError('Unsupported computational search method ', search)


    def _init_section(self, X_glob, X_loc, coords, constant):
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        if constant:
            n_vars = n_glob + n_loc + 1
        else:
            n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            nn = 40 + 2 * n_vars
            sq_dists = squareform(pdist(coords))
            sort_dists = np.sort(sq_dists, axis=1)
            min_dists = sort_dists[:,nn-1]
            max_dists = sort_dists[:,-1]
            a = np.min(min_dists)/2.0
            c = np.max(max_dists)/2.0

        if a < self.bw_min:
            a = self.bw_min
        if c > self.bw_max and self.bw_max > 0:
            c = self.bw_max
        return a, c
