# Main GWR classes

__author__ = "Taylor Oshan Tayoshan@gmail.com"

import copy
import numpy as np
import numpy.linalg as la
from scipy.stats import t
from scipy.special import factorial
from itertools import combinations as combo
from pysal.model.spglm.family import Gaussian, Binomial, Poisson
from pysal.model.spglm.glm import GLM, GLMResults
from pysal.model.spglm.iwls import iwls, _compute_betas_gwr
from pysal.model.spglm.utils import cache_readonly
from .diagnostics import get_AIC, get_AICc, get_BIC, corr
from .kernels import *
from .summary import *
import multiprocessing as mp


class GWR(GLM):
    """
    Geographically weighted regression. Can currently estimate Gaussian,
    Poisson, and logistic models(built on a GLM framework). GWR object prepares
    model input. Fit method performs estimation and returns a GWRResults object.

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observatons; also used as calibration locations is
                    'points' is set to None

    y             : array
                    n*1, dependent variable

    X             : array
                    n*k, independent variable, exlcuding the constant

    bw            : scalar
                    bandwidth value consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    Sel_BW

    family        : family object
                    underlying probability model; provides
                    distribution-specific calculations

    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations;
                    only for Poisson models

    sigma2_v1     : boolean
                    specify form of corrected denominator of sigma squared to use for
                    model diagnostics; Acceptable options are:

                    'True':       n-tr(S) (defualt)
                    'False':     n-2(tr(S)+tr(S'S))

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept.

    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
    hat_matrix    : boolean
                    True to store full n by n hat matrix,
                    False to not store full hat matrix to minimize memory footprint (defalut).

    Attributes
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates used for
                    calibration locations

    y             : array
                    n*1, dependent variable

    X             : array
                    n*k, independent variable, exlcuding the constant

    bw            : scalar
                    bandwidth value consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    Sel_BW

    family        : family object
                    underlying probability model; provides
                    distribution-specific calculations

    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations

    sigma2_v1     : boolean
                    specify form of corrected denominator of sigma squared to use for
                    model diagnostics; Acceptable options are:

                    'True':       n-tr(S) (defualt)
                    'False':     n-2(tr(S)+tr(S'S))

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept

    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
                    
    hat_matrix    : boolean
                    True to store full n by n hat matrix,
                    False to not store full hat matrix to minimize memory footprint (defalut).

    n             : integer
                    number of observations

    k             : integer
                    number of independent variables

    mean_y        : float
                    mean of y

    std_y         : float
                    standard deviation of y

    fit_params    : dict
                    parameters passed into fit method to define estimation
                    routine

    points        : array-like
                    n*2, collection of n sets of (x,y) coordinates used for
                    calibration locations instead of all observations;
                    defaults to None unles specified in predict method

    P             : array
                    n*k, independent variables used to make prediction;
                    exlcuding the constant; default to None unless specified
                    in predict method

    exog_scale    : scalar
                    estimated scale using sampled locations; defualt is None
                    unless specified in predict method

    exog_resid    : array-like
                    estimated residuals using sampled locations; defualt is None
                    unless specified in predict method

    Examples
    --------
    #basic model calibration

    >>> import pysal.lib as ps
    >>> from mgwr.gwr import GWR
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    >>> model = GWR(coords, y, X, bw=90.000, fixed=False, kernel='bisquare')
    >>> results = model.fit()
    >>> print(results.params.shape)
    (159, 4)

    #predict at unsampled locations

    >>> index = np.arange(len(y))
    >>> test = index[-10:]
    >>> X_test = X[test]
    >>> coords_test = np.array(coords)[test]
    >>> model = GWR(coords, y, X, bw=94, fixed=False, kernel='bisquare')
    >>> results = model.predict(coords_test, X_test)
    >>> print(results.params.shape)
    (10, 4)

    """

    def __init__(self, coords, y, X, bw, family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}

        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.spherical = spherical
        self.hat_matrix = hat_matrix

    def _build_wi(self, i, bw):

        try:
            wi = Kernel(i, self.coords, bw, fixed=self.fixed,
                        function=self.kernel, points=self.points,
                        spherical=self.spherical).kernel
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def _local_fit(self, i):
        """
        Local fitting at location i.
        """
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  #local spatial weights

        if isinstance(self.family, Gaussian):
            betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
            predy = np.dot(self.X[i], betas)[0]
            resid = self.y[i] - predy
            influ = np.dot(self.X[i], inv_xtx_xt[:, i])
            w = 1

        elif isinstance(self.family, (Poisson, Binomial)):
            rslt = iwls(self.y, self.X, self.family, self.offset, None,
                        self.fit_params['ini_params'], self.fit_params['tol'],
                        self.fit_params['max_iter'], wi=wi)
            inv_xtx_xt = rslt[5]
            w = rslt[3][i][0]
            influ = np.dot(self.X[i], inv_xtx_xt[:, i]) * w
            predy = rslt[1][i]
            resid = self.y[i] - predy
            betas = rslt[0]

        if self.fit_params['lite']:
            return influ, resid, predy, betas.reshape(-1)
        else:
            Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si * w * w)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
            if not self.hat_matrix:
                Si = None
            return influ, resid, predy, betas.reshape(-1), w, Si, tr_STS_i, CCT

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls',
            lite=False, pool=None):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------

        ini_betas     : array, optional
                        k*1, initial coefficient values, including constant.
                        Default is None, which calculates initial values during
                        estimation.
        tol:            float, optional
                        Tolerence for estimation convergence.
                        Default is 1.0e-5.
        max_iter      : integer, optional
                        Maximum number of iterations if convergence not
                        achieved. Default is 20.
        solve         : string, optional
                        Technique to solve MLE equations.
                        Default is 'iwls', meaning iteratively (
                        re)weighted least squares.
        lite          : bool, optional
                        Whether to estimate a lightweight GWR that
                        computes the minimum diagnostics needed for
                        bandwidth selection (could speed up
                        bandwidth selection for GWR) or to estimate
                        a full GWR. Default is False.
        pool          : A multiprocessing Pool object to enable parallel fitting; default is None.

        Returns
        -------
                      :
                        If lite=False, return a GWRResult
                        instance; otherwise, return a GWRResultLite
                        instance.

        """
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite

        if solve.lower() == 'iwls':

            if self.points is None:
                m = self.y.shape[0]
            else:
                m = self.points.shape[0]

            if pool:
                rslt = pool.map(self._local_fit,
                                range(m))  #parallel using mp.Pool
            else:
                rslt = map(self._local_fit, range(m))  #sequential

            rslt_list = list(zip(*rslt))
            influ = np.array(rslt_list[0]).reshape(-1, 1)
            resid = np.array(rslt_list[1]).reshape(-1, 1)
            params = np.array(rslt_list[3])

            if lite:
                return GWRResultsLite(self, resid, influ, params)
            else:
                predy = np.array(rslt_list[2]).reshape(-1, 1)
                w = np.array(rslt_list[-4]).reshape(-1, 1)
                if self.hat_matrix:
                    S = np.array(rslt_list[-3])
                else:
                    S = None
                tr_STS = np.sum(np.array(rslt_list[-2]))
                CCT = np.array(rslt_list[-1])
                return GWRResults(self, params, predy, S, CCT, influ, tr_STS,
                                  w)

    def predict(self, points, P, exog_scale=None, exog_resid=None,
                fit_params={}):
        """
        Method that predicts values of the dependent variable at un-sampled
        locations

        Parameters
        ----------
        points        : array-like
                        n*2, collection of n sets of (x,y) coordinates used for
                        calibration prediction locations
        P             : array
                        n*k, independent variables used to make prediction;
                        exlcuding the constant
        exog_scale    : scalar
                        estimated scale using sampled locations; defualt is None
                        which estimates a model using points from "coords"
        exog_resid    : array-like
                        estimated residuals using sampled locations; defualt is None
                        which estimates a model using points from "coords"; if
                        given it must be n*1 where n is the length of coords
        fit_params    : dict
                        key-value pairs of parameters that will be passed into fit
                        method to define estimation routine; see fit method for more details

        """
        if (exog_scale is None) & (exog_resid is None):
            train_gwr = self.fit(**fit_params)
            self.exog_scale = train_gwr.scale
            self.exog_resid = train_gwr.resid_response
        elif (exog_scale is not None) & (exog_resid is not None):
            self.exog_scale = exog_scale
            self.exog_resid = exog_resid
        else:
            raise InputError('exog_scale and exog_resid must both either be'
                             'None or specified')
        self.points = points
        if self.constant:
            P = np.hstack([np.ones((len(P), 1)), P])
            self.P = P
        else:
            self.P = P
        gwr = self.fit(**fit_params)

        return gwr

    @cache_readonly
    def df_model(self):
        return None

    @cache_readonly
    def df_resid(self):
        return None


class GWRResults(GLMResults):
    """
    Basic class including common properties for all GWR regression models

    Parameters
    ----------
    model               : GWR object
                        pointer to GWR object with estimation parameters

    params              : array
                          n*k, estimated coefficients

    predy               : array
                          n*1, predicted y values

    S                   : array
                          n*n, hat matrix

    CCT                 : array
                          n*k, scaled variance-covariance matrix

    w                   : array
                          n*1, final weight used for iteratively re-weighted least
                          sqaures; default is None

    Attributes
    ----------
    model               : GWR Object
                          points to GWR object for which parameters have been
                          estimated

    params              : array
                          n*k, parameter estimates

    predy               : array
                          n*1, predicted value of y

    y                   : array
                          n*1, dependent variable

    X                   : array
                          n*k, independent variable, including constant

    family              : family object
                          underlying probability model; provides
                          distribution-specific calculations

    n                   : integer
                          number of observations

    k                   : integer
                          number of independent variables

    df_model            : integer
                          model degrees of freedom

    df_resid            : integer
                          residual degrees of freedom

    offset              : array
                          n*1, the offset variable at the ith location.
                          For Poisson model this term is often the size of
                          the population at risk or the expected size of
                          the outcome in spatial epidemiology; Default is
                          None where Ni becomes 1.0 for all locations

    scale               : float
                          sigma squared used for subsequent computations

    w                   : array
                          n*1, final weights from iteratively re-weighted least
                          sqaures routine

    resid_response      : array
                          n*1, residuals of the repsonse

    resid_ss            : scalar
                          residual sum of sqaures

    W                   : array
                          n*n; spatial weights for each observation from each
                          calibration point

    S                   : array
                          n*n, hat matrix

    CCT                 : array
                          n*k, scaled variance-covariance matrix

    ENP                 : scalar
                          effective number of paramters, which depends on
                          sigma2

    tr_S                : float
                          trace of S (hat) matrix

    tr_STS              : float
                          trace of STS matrix

    y_bar               : array
                          n*1, weighted mean value of y

    TSS                 : array
                          n*1, geographically weighted total sum of squares

    RSS                 : array
                          n*1, geographically weighted residual sum of squares

    R2                  : float
                          R-squared for the entire model (1- RSS/TSS)
                          
    adj_R2              : float
                          adjusted R-squared for the entire model
                          
    aic                 : float
                          Akaike information criterion

    aicc                : float
                          corrected Akaike information criterion to account
                          to account for model complexity (smaller
                          bandwidths)

    bic                 : float
                          Bayesian information criterio

    localR2             : array
                          n*1, local R square

    sigma2              : float
                          sigma squared (residual variance) that has been
                          corrected to account for the ENP

    std_res             : array
                          n*1, standardised residuals

    bse                 : array
                          n*k, standard errors of parameters (betas)

    influ               : array
                          n*1, leading diagonal of S matrix

    CooksD              : array
                          n*1, Cook's D

    tvalues             : array
                          n*k, local t-statistics

    adj_alpha           : array
                          3*1, corrected alpha values to account for multiple
                          hypothesis testing for the 90%, 95%, and 99% confidence
                          levels; tvalues with an absolute value larger than the
                          corrected alpha are considered statistically
                          significant.

    deviance            : array
                          n*1, local model deviance for each calibration point

    resid_deviance      : array
                          n*1, local sum of residual deviance for each
                          calibration point

    llf                 : scalar
                          log-likelihood of the full model; see
                          pysal.contrib.glm.family for damily-sepcific
                          log-likelihoods

    pDev                : float
                          local percent of deviation accounted for; analogous to
                          r-squared for GLM's
                          
    D2                  : float
                          percent deviance explained for GLM, equivaleng to R2 for
                          Gaussian.
                          
    adj_D2              : float
                          adjusted percent deviance explained, equivaleng to adjusted
                          R2 for Gaussian.

    mu                  : array
                          n*, flat one dimensional array of predicted mean
                          response value from estimator

    fit_params          : dict
                          parameters passed into fit method to define estimation
                          routine

    predictions         : array
                          p*1, predicted values generated by calling the GWR
                          predict method to predict dependent variable at
                          unsampled points ()
    """

    def __init__(self, model, params, predy, S, CCT, influ, tr_STS=None,
                 w=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.tr_STS = tr_STS
        self.influ = influ
        self.CCT = self.cov_params(CCT, model.exog_scale)
        self._cache = {}

    @cache_readonly
    def W(self):
        W = np.array(
            [self.model._build_wi(i, self.model.bw) for i in range(self.n)])
        return W

    @cache_readonly
    def resid_ss(self):
        if self.model.points is not None:
            raise NotImplementedError('Not available for GWR prediction')
        else:
            u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self, scale=None):
        if isinstance(self.family, Gaussian):
            scale = self.sigma2
        else:
            scale = 1.0
        return scale

    def cov_params(self, cov, exog_scale=None):
        """
        Returns scaled covariance parameters

        Parameters
        ----------
        cov         : array
                      estimated covariance parameters

        Returns
        -------
        Scaled covariance parameters

        """
        if exog_scale is not None:
            return cov * exog_scale
        else:
            return cov * self.scale

    @cache_readonly
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        return np.sum(self.influ)

    @cache_readonly
    def ENP(self):
        """
        effective number of parameters

        Defaults to tr(s) as defined in :cite:`yu:2019`

        but can alternatively be based on 2tr(s) - tr(STS)

        and the form depends on the specification of sigma2
        """
        if self.model.sigma2_v1:
            return self.tr_S
        else:
            return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def y_bar(self):
        """
        weighted mean of y
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):
            w_i = np.reshape(self.model._build_wi(i, self.model.bw), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar

    @cache_readonly
    def TSS(self):
        """
        geographically weighted total sum of squares

        Methods: p215, (9.9)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * (self.y.reshape(
                               (-1, 1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):
        """
        geographically weighted residual sum of squares

        Methods: p215, (9.10)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        if self.model.points is not None:
            n = len(self.model.points)
            resid = self.model.exog_resid.reshape((-1, 1))
        else:
            n = self.n
            resid = self.resid_response.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))
        for i in range(n):
            RSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * resid**2)
        return RSS

    @cache_readonly
    def localR2(self):
        """
        local R square

        Methods: p215, (9.8)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        if isinstance(self.family, Gaussian):
            return (self.TSS - self.RSS) / self.TSS
        else:
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2(self):
        """
        residual variance

        if sigma2_v1 is True: only use n-tr(S) in denominator

        Methods: p214, (9.6) :cite:`fotheringham_geographically_2002`
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        and as defined in :cite:`yu:2019`

        if sigma2_v1 is False (v1v2): use n-2(tr(S)+tr(S'S)) in denominator

        Methods: p55 (2.16)-(2.18) :cite:`fotheringham_geographically_2002`
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        """
        if self.model.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
            # could be changed to SWSTW - nothing to test against
            return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def std_res(self):
        """
        standardized residuals

        Methods:  p215, (9.7) :cite:`fotheringham_geographically_2002`
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        return self.resid_response.reshape(
            (-1, 1)) / (np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        """
        standard errors of Betas

        Methods:  p215, (2.15) and (2.21) :cite:`fotheringham_geographically_2002`
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        return np.sqrt(self.CCT)

    @cache_readonly
    def cooksD(self):
        """
        Influence: leading diagonal of S Matrix

        Methods: p216, (9.11) :cite:`fotheringham_geographically_2002`
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        return self.std_res**2 * self.influ / (self.tr_S * (1.0 - self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1, 1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(
                2.0 * self.W * (y * np.log(y / (ybar * off)) -
                                (y - ybar * off)), axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1, 1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1, 1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(), self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1, 1))

    @cache_readonly
    def pDev(self):
        """
        Local percentage of deviance accounted for. Described in the GWR4
        manual. Equivalent to 1 - (deviance/null deviance)
        """
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance / self.deviance)

    @cache_readonly
    def adj_alpha(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Utility function to derive the critical t-value based on given alpha
        that are needed for hypothesis testing

        Parameters
        ----------
        alpha           : scalar
                          critical value to determine which tvalues are
                          associated with statistically significant parameter
                          estimates. Default to None in which case the adjusted
                          alpha value at the 95 percent CI is automatically
                          used.

        Returns
        -------
        critical        : scalar
                          critical t-val based on alpha
        """
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        """
        Utility function to set tvalues with an absolute value smaller than the
        absolute value of the alpha (critical) value to 0. If critical_t
        is supplied than it is used directly to filter. If alpha is provided
        than the critical t value will be derived and used to filter. If neither
        are critical_t nor alpha are provided, an adjusted alpha at the 95
        percent CI will automatically be used to define the critical t-value and
        used to filter. If both critical_t and alpha are supplied then the alpha
        value will be ignored.

        Parameters
        ----------
        critical_t      : scalar
                          critical t-value to determine whether parameters are
                          statistically significant

        alpha           : scalar
                          alpha value to determine which tvalues are
                          associated with statistically significant parameter
                          estimates

        Returns
        -------
        filtered       : array
                          n*k; new set of n tvalues for each of k variables
                          where absolute tvalues less than the absolute value of
                          alpha have been set to 0.
        """
        n = self.n
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0 * self.tr_S + self.tr_STS

    @cache_readonly
    def normalized_cov_params(self):
        return None

    @cache_readonly
    def resid_pearson(self):
        return None

    @cache_readonly
    def resid_working(self):
        return None

    @cache_readonly
    def resid_anscombe(self):
        return None

    @cache_readonly
    def pearson_chi2(self):
        return None

    @cache_readonly
    def llnull(self):
        return None

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)

    @cache_readonly
    def global_deviance(self):
        deviance = np.sum(self.family.resid_dev(self.y, self.mu)**2)
        return deviance

    @cache_readonly
    def D2(self):
        """
        Percentage of deviance explanied. Equivalent to 1 - (deviance/null deviance)
        """
        D2 = 1.0 - (self.global_deviance / self.null_deviance)
        return D2

    @cache_readonly
    def R2(self):
        """
        Global r-squared value for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.D2
        else:
            raise NotImplementedError('R2 only for Gaussian')

    @cache_readonly
    def adj_D2(self):
        """
        Adjusted percentage of deviance explanied.
        """
        adj_D2 = 1 - (1 - self.D2) * (self.n - 1) / (self.n - self.ENP - 1)
        return adj_D2

    @cache_readonly
    def adj_R2(self):
        """
        Adjusted global r-squared for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.adj_D2
        else:
            raise NotImplementedError('adjusted R2 only for Gaussian')

    @cache_readonly
    def aic(self):
        return get_AIC(self)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def bic(self):
        return get_BIC(self)

    @cache_readonly
    def pseudoR2(self):
        return None

    @cache_readonly
    def adj_pseudoR2(self):
        return None

    @cache_readonly
    def pvalues(self):
        return None

    @cache_readonly
    def conf_int(self):
        return None

    @cache_readonly
    def use_t(self):
        return None

    def local_collinearity(self):
        """
        Computes several indicators of multicollinearity within a geographically
        weighted design matrix, including:

        local correlation coefficients (n, ((p**2) + p) / 2)
        local variance inflation factors (VIF) (n, p-1)
        local condition number (n, 1)
        local variance-decomposition proportions (n, p)

        Returns four arrays with the order and dimensions listed above where n
        is the number of locations used as calibrations points and p is the
        number of explanatory variables. Local correlation coefficient and local
        VIF are not calculated for constant term.

        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = len(w)
        if self.model.constant:
            ncor = (((nvar - 1)**2 + (nvar - 1)) / 2) - (nvar - 1)
            jk = list(combo(range(1, nvar), 2))
        else:
            ncor = (((nvar)**2 + (nvar)) / 2) - nvar
            jk = list(combo(range(nvar), 2))
        corr_mat = np.ndarray((nrow, int(ncor)))
        if self.model.constant:
            vifs_mat = np.ndarray((nrow, nvar - 1))
        else:
            vifs_mat = np.ndarray((nrow, nvar))
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            wi = self.model._build_wi(i, self.model.bw)
            sw = np.sum(wi)
            wi = wi / sw
            tag = 0

            for j, k in jk:
                corr_mat[i, tag] = corr(np.cov(x[:, j], x[:, k],
                                               aweights=wi))[0][1]
                tag = tag + 1

            if self.model.constant:
                corr_mati = corr(np.cov(x[:, 1:].T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar - 1))))

            else:
                corr_mati = corr(np.cov(x.T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar))))

            xw = x * wi.reshape((nrow, 1))
            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]
            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return corr_mat, vifs_mat, local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Method to compute a Monte Carlo test of spatial variability for each
        estimated coefficient surface.

        WARNING: This test is very computationally demanding!

        Parameters
        ----------
        selector        : sel_bw object
                          should be the sel_bw object used to select a bandwidth
                          for the gwr model that produced the surfaces that are
                          being tested for spatial variation

        n_iters         : int
                          the number of Monte Carlo iterations to include for
                          the tests of spatial variability.

        seed            : int
                          optional parameter to select a custom seed to ensure
                          stochastic results are replicable. Default is none
                          which automatically sets the seed to 5536

        Returns
        -------

        p values        : list
                          a list of psuedo p-values that correspond to the model
                          parameter surfaces. Allows us to assess the
                          probability of obtaining the observed spatial
                          variation of a given surface by random chance.


        """
        temp_sel = copy.deepcopy(selector)
        temp_gwr = copy.deepcopy(self.model)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        fit_params = temp_gwr.fit_params
        search_params = temp_sel.search_params
        kernel = temp_gwr.kernel
        fixed = temp_gwr.fixed

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        for x in range(n_iters):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_bw = temp_sel.search(**search_params)
            temp_gwr.bw = temp_bw
            temp_gwr.coords = temp_coords
            temp_params = temp_gwr.fit(**fit_params).params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    @cache_readonly
    def predictions(self):
        P = self.model.P
        if P is None:
            raise TypeError('predictions only avaialble if predict'
                            'method is previously called on GWR model')
        else:
            predictions = np.sum(P * self.params, axis=1).reshape((-1, 1))
        return predictions

    def summary(self):
        """
        Print out GWR summary
        """
        summary = summaryModel(self) + summaryGLM(self) + summaryGWR(self)
        print(summary)
        return


class GWRResultsLite(object):
    """
    Lightweight GWR that computes the minimum diagnostics needed for bandwidth
    selection

    Parameters
    ----------
    model               : GWR object
                        pointer to GWR object with estimation parameters

    resid               : array
                        n*1, residuals of the repsonse

    influ               : array
                        n*1, leading diagonal of S matrix

    Attributes
    ----------
    tr_S                : float
                        trace of S (hat) matrix

    llf                 : scalar
                        log-likelihood of the full model; see
                        pysal.contrib.glm.family for damily-sepcific
                        log-likelihoods

    mu                  : array
                        n*, flat one dimensional array of predicted mean
                        response value from estimator

    resid_ss            : scalar
                          residual sum of sqaures

    """

    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.family = model.family
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.model = model
        self.params = params

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y, self.mu)

    @cache_readonly
    def mu(self):
        return self.y - self.resid_response

    @cache_readonly
    def predy(self):
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)


class MGWR(GWR):
    """
    Multiscale GWR estimation and inference.
    See :cite:`Fotheringham:2017` :cite:`yu:2019`.

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observatons; also used as calibration locations is
                    'points' is set to None

    y             : array
                    n*1, dependent variable

    X             : array
                    n*k, independent variable, exlcuding the constant

    selector      : sel_bw object
                    valid sel_bw object that has successfully called
                    the "search" method. This parameter passes on
                    information from GAM model estimation including optimal
                    bandwidths.

    family        : family object
                    underlying probability model; provides
                    distribution-specific calculations

    sigma2_v1     : boolean
                    specify form of corrected denominator of sigma squared to use for
                    model diagnostics; Acceptable options are:

                    'True':       n-tr(S) (defualt)
                    'False':     n-2(tr(S)+tr(S'S))

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept.

    spherical     : boolean
                    True for spherical coordinates (long-lat),
                    False for projected coordinates (defalut).
    hat_matrix    : boolean
                    True for computing and storing covariate-specific
                    hat matrices R (n,n,k) and model hat matrix S (n,n).
                    False (default) for computing MGWR inference on the fly.

    Attributes
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observatons; also used as calibration locations is
                    'points' is set to None

    y             : array
                    n*1, dependent variable

    X             : array
                    n*k, independent variable, exlcuding the constant

    selector      : sel_bw object
                    valid sel_bw object that has successfully called
                    the "search" method. This parameter passes on
                    information from GAM model estimation including optimal
                    bandwidths.

    bw            : array-like
                    collection of bandwidth values consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    Sel_BW with fb=True. Order of values should the same as
                    the order of columns associated with X

    family        : family object
                    underlying probability model; provides
                    distribution-specific calculations

    sigma2_v1     : boolean
                    specify form of corrected denominator of sigma squared to use for
                    model diagnostics; Acceptable options are:

                    'True':       n-tr(S) (defualt)
                    'False':     n-2(tr(S)+tr(S'S))

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept.

    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).

    n             : integer
                    number of observations

    k             : integer
                    number of independent variables

    mean_y        : float
                    mean of y

    std_y         : float
                    standard deviation of y

    fit_params    : dict
                    parameters passed into fit method to define estimation
                    routine

    W             : array-like
                    list of n*n arrays, spatial weights matrices for weighting all
                    observations from each calibration point: one for each
                    covariate (k)

    Examples
    --------

    #basic model calibration

    >>> import pysal.lib as ps
    >>> from mgwr.gwr import MGWR
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> fb = np.array(data.by_col('PctFB')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([fb, african_amer, rural])
    >>> X = (X - X.mean(axis=0)) / X.std(axis=0)
    >>> y = (y - y.mean(axis=0)) / y.std(axis=0)
    >>> selector = Sel_BW(coords, y, X, multi=True)
    >>> selector.search(multi_bw_min=[2])
    [92.0, 101.0, 136.0, 158.0]
    >>> model = MGWR(coords, y, X, selector, fixed=False, kernel='bisquare', sigma2_v1=True)
    >>> results = model.fit()
    >>> print(results.params.shape)
    (159, 4)

    """

    def __init__(self, coords, y, X, selector, sigma2_v1=True,
                 kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
        self.selector = selector
        self.bws = self.selector.bw[0]  #final set of bandwidth
        self.bws_history = selector.bw[1]  #bws history in backfitting
        self.bw_init = self.selector.bw_init  #initialization bandiwdth
        self.family = Gaussian(
        )  # manually set since we only support Gassian MGWR for now
        GWR.__init__(self, coords, y, X, self.bw_init, family=self.family,
                     sigma2_v1=sigma2_v1, kernel=kernel, fixed=fixed,
                     constant=constant, spherical=spherical,
                     hat_matrix=hat_matrix)
        self.selector = selector
        self.sigma2_v1 = sigma2_v1
        self.points = None
        self.P = None
        self.offset = None
        self.exog_resid = None
        self.exog_scale = None
        self_fit_params = None

    def _chunk_compute_R(self, chunk_id=0):
        """
        Compute MGWR inference by chunks to reduce memory footprint.
        """
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) *
                                   chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k))  #partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw_init).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  #n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                        chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        if self.hat_matrix:
            return ENP_j, CCT, pR
        return ENP_j, CCT

    def fit(self, n_chunks=1, pool=None):
        """
        Compute MGWR inference by chunk to reduce memory footprint.
        
        Parameters
        ----------

        n_chunks      : integer, optional
                        A number of chunks parameter to reduce memory usage. 
                        e.g. n_chunks=2 should reduce overall memory usage by 2.
        pool          : A multiprocessing Pool object to enable parallel fitting; default is None.
                        
        Returns
        -------
                      : MGWRResults
        """
        params = self.selector.params
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        try:
            from tqdm.autonotebook import tqdm  #progress bar
        except ImportError:

            def tqdm(x, total=0,
                     desc=''):  #otherwise, just passthrough the range
                return x

        if pool:
            self.n_chunks = pool._processes * n_chunks
            rslt = tqdm(
                pool.imap(self._chunk_compute_R, range(self.n_chunks)),
                total=self.n_chunks, desc='Inference')
        else:
            self.n_chunks = n_chunks
            rslt = map(self._chunk_compute_R,
                       tqdm(range(self.n_chunks), desc='Inference'))

        rslt_list = list(zip(*rslt))
        ENP_j = np.sum(np.array(rslt_list[0]), axis=0)
        CCT = np.sum(np.array(rslt_list[1]), axis=0)

        w = np.ones(self.n)
        if self.hat_matrix:
            R = np.hstack(rslt_list[2])
        else:
            R = None
        return MGWRResults(self, params, predy, CCT, ENP_j, w, R)

    def predict(self):
        '''
        Not implemented.
        '''
        raise NotImplementedError('N/A')


class MGWRResults(GWRResults):
    """
    Class including common properties for a MGWR model.

    Parameters
    ----------
    model               : MGWR object
                          pointer to MGWR object with estimation parameters

    params              : array
                          n*k, estimated coefficients

    predy               : array
                          n*1, predicted y values

    S                   : array
                          n*n, model hat matrix (if MGWR(hat_matrix=True))

    R                   : array
                          n*n*k, covariate-specific hat matrices (if MGWR(hat_matrix=True))

    CCT                 : array
                          n*k, scaled variance-covariance matrix

    w                   : array
                          n*1, final weight used for iteratively re-weighted least
                          sqaures; default is None

    Attributes
    ----------
    model               : GWR Object
                          points to GWR object for which parameters have been
                          estimated

    params              : array
                          n*k, parameter estimates

    predy               : array
                          n*1, predicted value of y

    y                   : array
                          n*1, dependent variable

    X                   : array
                          n*k, independent variable, including constant

    family              : family object
                          underlying probability model; provides
                          distribution-specific calculations

    n                   : integer
                          number of observations

    k                   : integer
                          number of independent variables

    df_model            : integer
                          model degrees of freedom

    df_resid            : integer
                          residual degrees of freedom

    scale               : float
                          sigma squared used for subsequent computations

    w                   : array
                          n*1, final weights from iteratively re-weighted least
                          sqaures routine

    resid_response      : array
                          n*1, residuals of the repsonse

    resid_ss            : scalar
                          residual sum of sqaures

    W                   : array-like
                          list of n*n arrays, spatial weights matrices for weighting all
                          observations from each calibration point: one for each
                          covariate (k)

    S                   : array
                          n*n, model hat matrix (if MGWR(hat_matrix=True))

    R                   : array
                          n*n*k, covariate-specific hat matrices (if MGWR(hat_matrix=True))

    CCT                 : array
                          n*k, scaled variance-covariance matrix

    ENP                 : scalar
                          effective number of paramters, which depends on
                          sigma2, for the entire model

    ENP_j               : array-like
                          effective number of paramters, which depends on
                          sigma2, for each covariate in the model

    adj_alpha           : array
                          3*1, corrected alpha values to account for multiple
                          hypothesis testing for the 90%, 95%, and 99% confidence
                          levels; tvalues with an absolute value larger than the
                          corrected alpha are considered statistically
                          significant.

    adj_alpha_j         : array
                          k*3, corrected alpha values to account for multiple
                          hypothesis testing for the 90%, 95%, and 99% confidence
                          levels; tvalues with an absolute value larger than the
                          corrected alpha are considered statistically
                          significant. A set of alpha calues is computed for
                          each covariate in the model.

    tr_S                : float
                          trace of S (hat) matrix

    tr_STS              : float
                          trace of STS matrix

    R2                  : float
                          R-squared for the entire model (1- RSS/TSS)
                          
    adj_R2              : float
                          adjusted R-squared for the entire model

    aic                 : float
                          Akaike information criterion

    aicc                : float
                          corrected Akaike information criterion to account
                          to account for model complexity (smaller
                          bandwidths)

    bic                 : float
                          Bayesian information criterio

    sigma2              : float
                          sigma squared (residual variance) that has been
                          corrected to account for the ENP

    std_res             : array
                          n*1, standardised residuals

    bse                 : array
                          n*k, standard errors of parameters (betas)

    influ               : array
                          n*1, leading diagonal of S matrix

    CooksD              : array
                          n*1, Cook's D

    tvalues             : array
                          n*k, local t-statistics

    llf                 : scalar
                          log-likelihood of the full model; see
                          pysal.contrib.glm.family for damily-sepcific
                          log-likelihoods

    mu                  : array
                          n*, flat one dimensional array of predicted mean
                          response value from estimator

    """

    def __init__(self, model, params, predy, CCT, ENP_j, w, R):
        """
        Initialize class
        """
        self.ENP_j = ENP_j
        self.R = R
        GWRResults.__init__(self, model, params, predy, None, CCT, None, w)
        if model.hat_matrix:
            self.S = np.sum(self.R, axis=2)
        self.predy = predy

    @cache_readonly
    def tr_S(self):
        return np.sum(self.ENP_j)

    @cache_readonly
    def W(self):
        Ws = []
        for bw_j in self.model.bws:
            W = np.array(
                [self.model._build_wi(i, bw_j) for i in range(self.n)])
            Ws.append(W)
        return Ws

    @cache_readonly
    def adj_alpha_j(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        :cite:`Silva:2016` : da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = np.array(self.ENP_j).reshape((-1, 1))
        p = 1.
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Utility function to derive the critial t-value based on given alpha
        that are needed for hypothesis testing

        Parameters
        ----------
        alpha           : scalar
                          critical value to determine which tvalues are
                          associated with statistically significant parameter
                          estimates. Default to None in which case the adjusted
                          alpha value at the 95 percent CI is automatically
                          used.

        Returns
        -------
        critical        : scalar
                          critical t-val based on alpha
        """
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        """
        Utility function to set tvalues with an absolute value smaller than the
        absolute value of the alpha (critical) value to 0. If critical_t
        is supplied than it is used directly to filter. If alpha is provided
        than the critical t value will be derived and used to filter. If neither
        are critical_t nor alpha are provided, an adjusted alpha at the 95
        percent CI will automatically be used to define the critical t-value and
        used to filter. If both critical_t and alpha are supplied then the alpha
        value will be ignored.

        Parameters
        ----------
        critical        : scalar
                          critical t-value to determine whether parameters are
                          statistically significant

        alpha           : scalar
                          alpha value to determine which tvalues are
                          associated with statistically significant parameter
                          estimates

        Returns
        -------
        filtered       : array
                          n*k; new set of n tvalues for each of k variables
                          where absolute tvalues less than the absolute value of
                          alpha have been set to 0.
        """
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def RSS(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def TSS(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def localR2(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def y_bar(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def predictions(self):
        raise NotImplementedError('Not yet implemented for MGWR')

    def local_collinearity(self):
        """
        Computes several indicators of multicollinearity within a geographically
        weighted design matrix, including:

        local condition number (n, 1)
        local variance-decomposition proportions (n, p)

        Returns four arrays with the order and dimensions listed above where n
        is the number of locations used as calibrations points and p is the
        nubmer of explanatory variables

        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = self.n
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            xw = np.zeros((x.shape))
            for j in range(nvar):
                wi = w[j][i]
                sw = np.sum(wi)
                wi = wi / sw
                xw[:, j] = x[:, j] * wi

            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]

            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Method to compute a Monte Carlo test of spatial variability for each
        estimated coefficient surface.

        WARNING: This test is very computationally demanding!

        Parameters
        ----------
        selector        : sel_bw object
                          should be the sel_bw object used to select a bandwidth
                          for the gwr model that produced the surfaces that are
                          being tested for spatial variation

        n_iters         : int
                          the number of Monte Carlo iterations to include for
                          the tests of spatial variability.

        seed            : int
                          optional parameter to select a custom seed to ensure
                          stochastic results are replicable. Default is none
                          which automatically sets the seed to 5536

        Returns
        -------

        p values        : list
                          a list of psuedo p-values that correspond to the model
                          parameter surfaces. Allows us to assess the
                          probability of obtaining the observed spatial
                          variation of a given surface by random chance.


        """
        temp_sel = copy.deepcopy(selector)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        search_params = temp_sel.search_params

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        for x in range(n_iters):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_sel.search(**search_params)
            temp_params = temp_sel.params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    def summary(self):
        """
        Print out MGWR summary
        """
        summary = summaryModel(self) + summaryGLM(self) + summaryMGWR(self)
        print(summary)
        return
