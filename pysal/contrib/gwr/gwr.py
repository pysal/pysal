#Main GWR classes

#Offset does not yet do anyhting and needs to be implemented

__author__ = "Taylor Oshan Tayoshan@gmail.com"

import numpy as np
import numpy.linalg as la
from scipy.stats import t
from kernels import *
from diagnostics import get_AIC, get_AICc, get_BIC
import pysal.spreg.user_output as USER
from pysal.contrib.glm.family import Gaussian, Binomial, Poisson
from pysal.contrib.glm.glm import GLM, GLMResults
from pysal.contrib.glm.iwls import iwls
from pysal.contrib.glm.utils import cache_readonly

fk = {'gaussian': fix_gauss, 'bisquare': fix_bisquare, 'exponential': fix_exp}
ak = {'gaussian': adapt_gauss, 'bisquare': adapt_bisquare, 'exponential': adapt_exp}

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
                        specify sigma squared, True to use n as denominator;
                        default is False which uses n-k

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
                        specify sigma squared, True to use n as denominator;
                        default is False which uses n-k

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

        W             : array
                        n*n, spatial weights matrix for weighting all
                        observations from each calibration point
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

    >>> import pysal
    >>> from pysal.contrib.gwr.gwr import GWR
    >>> data = pysal.open(pysal.examples.get_path('GData_utm.csv'))
    >>> coords = zip(data.bycol('X'), data.by_col('Y')) 
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    >>> model = GWR(coords, y, X, bw=90.000, fixed=False, kernel='bisquare')
    >>> results = model.fit()
    >>> print results.params.shape
    (159, 4)

    #predict at unsampled locations
    
    >>> index = np.arange(len(self.y))
    >>> test = index[-10:]
    >>> X_test = X[test]
    >>> coords_test = list(coords[test])
    >>> model = GWR(coords, y, X, bw=94, fixed=False, kernel='bisquare')
    >>> results = model.predict(coords_test, X_test)
    >>> print results.params.shape
    (10, 4)

    """
    def __init__(self, coords, y, X, bw, family=Gaussian(), offset=None,
            sigma2_v1=False, kernel='bisquare', fixed=False, constant=True):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = coords
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}
        self.W = self._build_W(fixed, kernel, coords, bw)
        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None

    def _build_W(self, fixed, kernel, coords, bw, points=None):
        if fixed:
            try:
                W = fk[kernel](coords, bw, points)
            except:
                raise TypeError('Unsupported kernel function  ', kernel)
        else:
            try:
                W = ak[kernel](coords, bw, points)
            except:
                raise TypeError('Unsupported kernel function  ', kernel)

        return W

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls'):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------

        ini_betas     : array
                        k*1, initial coefficient values, including constant.
                        Default is None, which calculates initial values during
                        estimation
        tol:            float
                        Tolerence for estimation convergence
        max_iter      : integer
                        Maximum number of iterations if convergence not
                        achieved
        solve         : string
                        Technique to solve MLE equations.
                        'iwls' = iteratively (re)weighted least squares (default)
        """
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']= solve
        if solve.lower() == 'iwls':
            m = self.W.shape[0]
            params = np.zeros((m, self.k))
            predy = np.zeros((m, 1))
            v = np.zeros((m, 1))
            w = np.zeros((m, 1))
            z = np.zeros((m, self.n))
            S = np.zeros((m, self.n))
            R = np.zeros((m, self.n))
            CCT = np.zeros((m, self.k))
            #f = np.zeros((n, n))
            p = np.zeros((m, 1))
            for i in range(m):
                wi = self.W[i].reshape((-1,1))
                rslt = iwls(self.y, self.X, self.family, self.offset, None,
                ini_params, tol, max_iter, wi=wi)
                params[i,:] = rslt[0].T
                predy[i] = rslt[1][i]
                v[i] = rslt[2][i]
                w[i] = rslt[3][i]
                z[i] = rslt[4].flatten()
                R[i] = np.dot(self.X[i], rslt[5])
                ri = np.dot(self.X[i], rslt[5])
                S[i] = ri*np.reshape(rslt[4].flatten(), (1,-1))
                #dont need unless f is explicitly passed for
                #prediction of non-sampled points
                #cf = rslt[5] - np.dot(rslt[5], f)
                #CCT[i] = np.diag(np.dot(cf, cf.T/rslt[3]))
                CCT[i] = np.diag(np.dot(rslt[5], rslt[5].T))
            S = S * (1.0/z)
        return GWRResults(self, params, predy, S, CCT, w)

    def predict(self, points, P, exog_scale=None, exog_resid=None, fit_params={}):
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
            P = np.hstack([np.ones((len(P),1)), P])
            self.P = P
        else:
            self.P = P
        self.W = self._build_W(self.fixed, self.kernel, self.coords, self.bw, points)
        gwr = self.fit(**fit_params)

        return gwr

    @cache_readonly
    def df_model(self):
        raise NotImplementedError('Only computed for fitted model in GWRResults')

    @cache_readonly
    def df_resid(self):
        raise NotImplementedError('Only computed for fitted model in GWRResults')

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

        w                   : array
                              n*1, final weight used for iteratively re-weighted least
                              sqaures; default is None

        S                   : array
                              n*n, hat matrix

        CCT                 : array
                              n*k, scaled variance-covariance matrix

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

        tr_S                : float
                              trace of S (hat) matrix

        tr_STS              : float
                              trace of STS matrix

        tr_SWSTW            : float
                              trace of weighted STS matrix; weights are those output
                              from iteratively weighted least sqaures (not spatial
                              weights)

        y_bar               : array
                              n*1, weighted mean value of y

        TSS                 : array
                              n*1, geographically weighted total sum of squares

        RSS                 : array
                              n*1, geographically weighted residual sum of squares

        localR2             : array
                              n*1, local R square

        sigma2_v1           : float
                              sigma squared, use (n-v1) as denominator

        sigma2_v1v2         : float
                              sigma squared, use (n-2v1+v2) as denominator

        sigma2_ML           : float
                              sigma squared, estimated using ML

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

        mu                  : array
                              n*, flat one dimensional array of predicted mean
                              response value from estimator

        fit_params          : dict
                              parameters passed into fit method to define estimation
                              routine
    """
    def __init__(self, model, params, predy, S, CCT, w=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.W = model.W
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.CCT = self.cov_params(CCT, model.exog_scale)
        self._cache = {}

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
            if self.model.sigma2_v1:
                scale = self.sigma2_v1
            else:
                scale = self.sigma2_v1v2
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
          return cov*exog_scale
        else:
            return cov*self.scale

    @cache_readonly
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        return np.trace(self.S*self.w)

    @cache_readonly
    def tr_STS(self):
        """
        trace of STS matrix
        """
        return np.trace(np.dot(self.S.T*self.w,self.S*self.w))

    @cache_readonly
    def y_bar(self):
        """
        weighted mean of y
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        off = self.offset.reshape((-1,1))
        arr_ybar = np.zeros(shape=(self.n,1))
        for i in range(n):
            w_i= np.reshape(np.array(self.W[i]), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1,1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i*off)
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
        TSS = np.zeros(shape=(n,1))
        for i in range(n):
            TSS[i] = np.sum(np.reshape(np.array(self.W[i]), (-1,1)) *
                (self.y.reshape((-1,1)) - self.y_bar[i])**2)
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
            resid = self.model.exog_resid.reshape((-1,1))
        else:
            n = self.n
            resid = self.resid_response.reshape((-1,1))
        RSS = np.zeros(shape=(n,1))
        for i in range(n):
            RSS[i] = np.sum(np.reshape(np.array(self.W[i]), (-1,1))
                * resid**2)
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
            return (self.TSS - self.RSS)/self.TSS
        else:
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2_v1(self):
        """
        residual variance

        Methods: p214, (9.6),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        only use v1
        """
        return (self.resid_ss/(self.n-self.tr_S))

    @cache_readonly
    def sigma2_v1v2(self):
        """
        residual variance

        Methods: p55 (2.16)-(2.18)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.

        use v1 and v2 #used in GWR4
        """
        if isinstance(self.family, (Poisson, Binomial)):
            return self.resid_ss/(self.n - 2.0*self.tr_S +
                self.tr_STS) #could be changed to SWSTW - nothing to test against
        else:
            return self.resid_ss/(self.n - 2.0*self.tr_S +
                self.tr_STS) #could be changed to SWSTW - nothing to test against
    @cache_readonly
    def sigma2_ML(self):
        """
        residual variance

        Methods: maximum likelihood
        """
        return self.resid_ss/self.n

    @cache_readonly
    def std_res(self):
        """
        standardized residuals

        Methods:  p215, (9.7)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        return self.resid_response.reshape((-1,1))/(np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        """
        standard errors of Betas

        Methods:  p215, (2.15) and (2.21)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        """
        return np.sqrt(self.CCT)

    @cache_readonly
    def influ(self):
        """
        Influence: leading diagonal of S Matrix
        """
        return np.reshape(np.diag(self.S),(-1,1))

    @cache_readonly
    def cooksD(self):
        """
        Influence: leading diagonal of S Matrix

        Methods: p216, (9.11),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying
        relationships.
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        return self.std_res**2 * self.influ / (self.tr_S * (1.0-self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1,1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(2.0*self.W*(y*np.log(y/(ybar*off))-(y-ybar*off)),axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1,1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1,1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(),self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1,1))

    @cache_readonly
    def pDev(self):
        """
        Local percentage of deviance accounted for. Described in the GWR4
        manual. Equivalent to 1 - (deviance/null deviance)
        """
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance/self.deviance)

    @cache_readonly
    def adj_alpha(self):
        """
        Corrected alpha (critical) values to account for multiple testing during hypothesis
        testing. Includes corrected value for 90% (.1), 95% (.05), and 99%
        (.01) confidence levels. Correction comes from:

        da Silva, A. R., & Fotheringham, A. S. (2015). The Multiple Testing Issue in
        Geographically Weighted Regression. Geographical Analysis.

        """
        alpha = np.array([.1, .05, .001])
        pe = (2.0 * self.tr_S) - self.tr_STS
        p = self.k
        return (alpha*p)/pe

    def filter_tvals(self, alpha):
        """
        Utility function to set tvalues with an absolute value smaller than the
        absolute value of the alpha (critical) value to 0

        Parameters
        ----------
        alpha           : scalar
                          critical value to determine which tvalues are
                          associated with statistically significant parameter
                          estimates

        Returns
        -------
        filtered       : array
                          n*k; new set of n tvalues for each of k variables
                          where absolute tvalues less than the absolute value of
                          alpha have been set to 0.
        """
        alpha = np.abs(alpha)/2.0
        n = self.n
        critical = t.ppf(1-alpha, n-1)
        subset = (self.tvalues < critical) & (self.tvalues > -1.0*critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0*self.tr_S + self.tr_STS

    @cache_readonly
    def normalized_cov_params(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def resid_pearson(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def resid_working(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def resid_anscombe(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def pearson_chi2(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def null(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def llnull(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def null_deviance(self):
        raise NotImplementedError('Not implemented for GWR')

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
    def D2(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def adj_D2(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def pseudoR2(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def adj_pseudoR2(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def pvalues(self):
        raise NotImplementedError('Not implemented for GWR')

    @cache_readonly
    def predictions(self):
        P = self.model.P
        if P is None:
            raise NotImplementedError('predictions only avaialble if predict'
            'method called on GWR model')
        else:
            predictions = np.sum(P*self.params, axis=1).reshape((-1,1))
        return predictions
