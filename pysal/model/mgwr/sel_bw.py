# GWR Bandwidth selection class

# x_glob parameter does not yet do anything; it is for semiparametric

__author__ = "Taylor Oshan Tayoshan@gmail.com"


from pysal.model.spreg import user_output as USER
import numpy as np
import multiprocessing as mp
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from pysal.model.spglm.family import Gaussian, Poisson, Binomial
from .kernels import Kernel
from .gwr import GWR
from .search import golden_section, equal_interval, multi_bw
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV

getDiag = {"AICc": get_AICc, "AIC": get_AIC, "BIC": get_BIC, "CV": get_CV}


class Sel_BW(object):
    """
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection

    :cite:`fotheringham_geographically_2002`: Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
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
    family         : family object/instance, optional
                     underlying probability model: Gaussian(), Poisson(),
                     Binomial(). Default is Gaussian().
    offset         : array
                     n*1, the offset variable at the ith location. For Poisson model
                     this term is often the size of the population at risk or
                     the expected size of the outcome in spatial epidemiology
                     Default is None where Ni becomes 1.0 for all locations
    kernel         : string, optional
                     kernel function: 'gaussian', 'bisquare', 'exponential'.
                     Default is 'bisquare'.
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    spherical      : boolean
                     True for shperical coordinates (long-lat),
                     False for projected coordinates (defalut).

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
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search_method  : string
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
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    offset         : array
                     n*1, the offset variable at the ith location. For Poisson model
                     this term is often the size of the population at risk or
                     the expected size of the outcome in spatial epidemiology
                     Default is None where Ni becomes 1.0 for all locations
    spherical      : boolean
                     True for shperical coordinates (long-lat),
                     False for projected coordinates (defalut).
    search_params  : dict
                     stores search arguments
    int_score      : boolan
                     True if adaptive bandwidth is being used and bandwdith
                     selection should be discrete. False
                     if fixed bandwidth is being used and bandwidth does not have
                     to be discrete.
    bw             : scalar or array-like
                     Derived optimal bandwidth(s). Will be a scalar for GWR
                     (multi=False) and a list of scalars for MGWR (multi=True)
                     with one bandwidth for each covariate.
    S              : array
                     n*n, hat matrix derived from the iterative backfitting
                     algorthim for MGWR during bandwidth selection
    R              : array
                     n*n*k, partial hat matrices derived from the iterative
                     backfitting algoruthm for MGWR during bandwidth selection.
                     There is one n*n matrix for each of the k covariates.
    params         : array
                     n*k, calibrated parameter estimates for MGWR based on the
                     iterative backfitting algorithm - computed and saved here to
                     avoid having to do it again in the MGWR object.

    Examples
    --------

    >>> import pysal.lib as ps
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    
    Golden section search AICc - adaptive bisquare

    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print(bw)
    93.0

    Golden section search AIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
    >>> print(bw)
    50.0

    Golden section search BIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
    >>> print(bw)
    62.0

    Golden section search CV - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
    >>> print(bw)
    68.0

    Interval AICc - fixed bisquare

    >>> sel = Sel_BW(coords, y, X, fixed=True)
    >>> bw = sel.search(search_method='interval', bw_min=211001.0, bw_max=211035.0, interval=2)
    >>> print(bw)
    211025.0

    """

    def __init__(
        self,
        coords,
        y,
        X_loc,
        X_glob=None,
        family=Gaussian(),
        offset=None,
        kernel="bisquare",
        fixed=False,
        multi=False,
        constant=True,
        spherical=False,
    ):
        self.coords = np.array(coords)
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family = family
        self.fixed = fixed
        self.kernel = kernel
        if offset is None:
            self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self.search_params = {}

    def search(
        self,
        search_method="golden_section",
        criterion="AICc",
        bw_min=None,
        bw_max=None,
        interval=0.0,
        tol=1.0e-6,
        max_iter=200,
        init_multi=None,
        tol_multi=1.0e-5,
        rss_score=False,
        max_iter_multi=200,
        multi_bw_min=[None],
        multi_bw_max=[None],
        bws_same_times=3,
        pool=None,
        verbose=False,
    ):
        """
        Method to select one unique bandwidth for a gwr model or a
        bandwidth vector for a mgwr model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search_method  : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        multi_bw_min   : list 
                         min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept
        multi_bw_max   : list
                         max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol
        init_multi     : float
                         None (default) to initialize MGWR with a bandwidth
                         derived from GWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.
        tol_multi      : convergence tolerence for the multiple bandwidth
                         backfitting algorithm; a larger tolerance may stop the
                         algorith faster though it may result in a less optimal
                         model
        max_iter_multi : max iterations if no convergence to tol for multiple
                         bandwidth backfitting algorithm
        rss_score      : True to use the residual sum of sqaures to evaluate
                         each iteration of the multiple bandwidth backfitting
                         routine and False to use a smooth function; default is
                         False
        bws_same_times : If bandwidths keep the same between iterations for
                         bws_same_times (default 3) in backfitting, then use the
                         current set of bandwidths as final bandwidths.
        pool           : A multiprocessing Pool object to enbale parallel fitting;
                         default is None
        verbose        : Boolean
                         If true, bandwidth searching history is printed out; default is False.

        Returns
        -------
        bw             : scalar or array
                         optimal bandwidth value or values; returns scalar for
                         multi=False and array for multi=True; ordering of bandwidths
                         matches the ordering of the covariates (columns) of the
                         designs matrix, X
        """
        k = self.X_loc.shape[1]
        if self.constant:  # k is the number of covariates
            k += 1
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bws_same_times = bws_same_times
        self.pool = pool
        self.verbose = verbose

        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min * k
        else:
            raise AttributeError(
                "multi_bw_min must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept"
            )

        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max * k
        else:
            raise AttributeError(
                "multi_bw_max must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept"
            )

        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_multi = init_multi
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params["search_method"] = search_method
        self.search_params["criterion"] = criterion
        self.search_params["bw_min"] = bw_min
        self.search_params["bw_max"] = bw_max
        self.search_params["interval"] = interval
        self.search_params["tol"] = tol
        self.search_params["max_iter"] = max_iter
        # self._check_min_max()

        self.int_score = not self.fixed

        if self.multi:
            self._mbw()
            self.params = self.bw[3]  # params n by k
            self.bw_init = self.bw[-1]  # scalar, optimal bw from initial gwr model
        else:
            self._bw()

        self.pool = None
        return self.bw[0]

    def _bw(self):
        gwr_func = lambda bw: getDiag[self.criterion](
            GWR(
                self.coords,
                self.y,
                self.X_loc,
                bw,
                family=self.family,
                kernel=self.kernel,
                fixed=self.fixed,
                constant=self.constant,
                offset=self.offset,
                spherical=self.spherical,
            ).fit(lite=True, pool=self.pool)
        )

        self._optimized_function = gwr_func

        if self.search_method == "golden_section":
            a, c = self._init_section(
                self.X_glob, self.X_loc, self.coords, self.constant
            )
            delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(
                a,
                c,
                delta,
                gwr_func,
                self.tol,
                self.max_iter,
                self.int_score,
                self.verbose,
            )
        elif self.search_method == "interval":
            self.bw = equal_interval(
                self.bw_min,
                self.bw_max,
                self.interval,
                gwr_func,
                self.int_score,
                self.verbose,
            )
        elif self.search_method == "scipy":
            self.bw_min, self.bw_max = self._init_section(
                self.X_glob, self.X_loc, self.coords, self.constant
            )
            if self.bw_min == self.bw_max:
                raise Exception(
                    "Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer."
                )
            self._optimize_result = minimize_scalar(
                gwr_func, bounds=(self.bw_min, self.bw_max), method="bounded"
            )
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError(
                "Unsupported computational search method ", self.search_method
            )

    def _mbw(self):
        y = self.y
        if self.constant:
            X = USER.check_constant(self.X_loc)
        else:
            X = self.X_loc
        n, k = X.shape
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
        spherical = self.spherical
        coords = self.coords
        search_method = self.search_method
        criterion = self.criterion
        bw_min = self.bw_min
        bw_max = self.bw_max
        multi_bw_min = self.multi_bw_min
        multi_bw_max = self.multi_bw_max
        interval = self.interval
        tol = self.tol
        max_iter = self.max_iter
        bws_same_times = self.bws_same_times

        def gwr_func(y, X, bw):
            return GWR(
                coords,
                y,
                X,
                bw,
                family=family,
                kernel=kernel,
                fixed=fixed,
                offset=offset,
                constant=False,
                spherical=self.spherical,
                hat_matrix=False,
            ).fit(lite=True, pool=self.pool)

        def bw_func(y, X):
            selector = Sel_BW(
                coords,
                y,
                X,
                X_glob=[],
                family=family,
                kernel=kernel,
                fixed=fixed,
                offset=offset,
                constant=False,
                spherical=self.spherical,
            )
            return selector

        def sel_func(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(
                search_method=search_method,
                criterion=criterion,
                bw_min=bw_min,
                bw_max=bw_max,
                interval=interval,
                tol=tol,
                max_iter=max_iter,
                pool=self.pool,
                verbose=False,
            )

        self.bw = multi_bw(
            self.init_multi,
            y,
            X,
            n,
            k,
            family,
            self.tol_multi,
            self.max_iter_multi,
            self.rss_score,
            gwr_func,
            bw_func,
            sel_func,
            multi_bw_min,
            multi_bw_max,
            bws_same_times,
            verbose=self.verbose,
        )

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
            sq_dists = pdist(coords)
            a = np.min(sq_dists) / 2.0
            c = np.max(sq_dists) * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None:
            c = self.bw_max

        return a, c
