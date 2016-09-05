
import numpy as np
import numpy.linalg as la
from pysal.spreg.utils import RegressionPropsY, spdot
import pysal.spreg.user_output as USER
from utils import cache_readonly
from base import LikelihoodModelResults
import family
from iwls import iwls

__all__ = ['GLM']

class GLM(RegressionPropsY):
    """
    Generalised linear models. Can currently estimate Guassian, Poisson and
    Logisitc regression coefficients. GLM object prepares model input and fit
    method performs estimation which then returns a GLMResults object.

    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        X             : array
                        n*k, independent variable, exlcuding the constant.
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'Binomial'
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        this term is often the size of the population at risk or
                        the expected size of the outcome in spatial epidemiology.
                        Default is None where Ni becomes 1.0 for all locations.
        y_fix         : array
                        n*1, the fix intercept value of y

    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        X             : array
                        n*k, independent variable, including constant.
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'logistic'
        n             : integer
                        Number of observations
        k             : integer
                        Number of independent variables
        df_model      : float
                        k-1, where k is the number of variables (including
                        intercept)
        df_residual   : float
                        observations minus variables (n-k)
        mean_y        : float
                        Mean of y
        std_y         : float
                        Standard deviation of y
        fit_params     : dict
                        Parameters passed into fit method to define estimation
                        routine.

    Examples
    --------
    >>> from pysal.contrib.glm.glm import GLM
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> self.y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> self.X = np.array(X).T
    >>> model = GLM(self.y, self.X, family=Gaussian())
    >>> results = model.fit()
    >>> results.params
    [ 46.42818268,   0.62898397, -0.48488854]

    """
    def __init__(self, y, X, family=family.Gaussian(), offset=None, y_fix = None,
            constant=True):
        """
        Initialize class
        """
        self.n = USER.check_arrays(y, X)
        USER.check_y(y, self.n)
        self.y = y
        if constant:
            self.X = USER.check_constant(X)
        else:
            self.X = X
        self.family = family
        self.k = self.X.shape[1]
        self.df_model = self.X.shape[1] - 1
        self.df_resid = self.n - self.df_model - 1
        if offset is None:
            self.offset = np.ones(shape=(self.n,1))
        else:
            self.offset = offset * 1.0
        if y_fix is None:
	        self.y_fix = np.zeros(shape=(self.n,1))
        else:
	        self.y_fix = y_fix
        self.fit_params = {}

    def fit(self, ini_betas=None, tol=1.0e-6, max_iter=200, solve='iwls'):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------

        ini_betas     : array
                        k*1, initial coefficient values, including constant.
                        Default is None, which calculates initial values during
                        estimation.
        tol:            float
                        Tolerence for estimation convergence.
        max_iter       : integer
                        Maximum number of iterations if convergence not
                        achieved.
        solve         :string
                       Technique to solve MLE equations.
                       'iwls' = iteratively (re)weighted least squares (default)
        """
        self.fit_params['ini_betas'] = ini_betas
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']=solve
        if solve.lower() == 'iwls':
            params, predy, w, n_iter = iwls(self.y, self.X, self.family, self.offset, 
                    self.y_fix, ini_betas, tol, max_iter)
            self.fit_params['n_iter'] = n_iter
        return GLMResults(self, params.flatten(), predy, w)


class GLMResults(LikelihoodModelResults):
    """
    Results of estimated GLM and diagnostics.

    Parameters
    ----------
        model         : GLM object
                        Pointer to GLM object with estimation parameters.
        params         : array
                        k*1, estimared coefficients
        mu         : array
                        n*1, predicted y values.
        w             : array
                        n*1, final weight used for iwls

    Attributes
    ----------
        model         : GLM Object
                        Points to GLM object for which parameters have been
                        estimated.
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'Logistic'
        n             : integer
                        Number of observations
        k             : integer
                        Number of independent variables
        df_model      : float
                        k-1, where k is the number of variables (including
                        intercept)
        df_residual   : float
                        observations minus variables (n-k)
        fit_params    : dict
                        parameters passed into fit method to define estimation
                        routine.
        scale         : float
                        sigma squared used for subsequent computations.
        params         : array
                        n*k, estimared beta coefficients
        w             : array
                        n*1, final weight values of x
        mu            : array
                        n*1, predicted value of y (i.e., fittedvalues)
        cov_params    : array
                        Variance covariance matrix (kxk) of betas
        bse           : array
                        k*1, standard errors of betas
        pvalues       : array
                        k*1, two-tailed pvalues of parameters
        tvalues       : array
                        k*1, the tvalues of the standard errors
        null          : array
                        n*1, predicted values of y for null model
        deviance      : float
                        value of the deviance function evalued at params;
                        see family.py for distribution-specific deviance
        null_deviance : float
                        value of the deviance function for the model fit with
                        a constant as the only regressor
        llf           : float
                        value of the loglikelihood function evalued at params;
                        see family.py for distribution-specific loglikelihoods
        llnull       : float
                        value of log-likelihood function evaluated at null
        aic           : float 
                        AIC
        bic           : float 
                        BIC
        D2            : float
                        percent deviance explained
        adj_D2        : float
                        adjusted percent deviance explained
        pseudo_R2       : float
                        McFadden's pseudo R2  (coefficient of determination) 
        adj_pseudoR2    : float
                        adjusted McFadden's pseudo R2
        resid_response          : array
                                  response residuals; defined as y-mu
        resid_pearson           : array
                                  Pearson residuals; defined as (y-mu)/sqrt(VAR(mu))
                                  where VAR is the distribution specific variance
                                  function; see family.py and varfuncs.py for more information.
        resid_working           : array
                                  Working residuals; the working residuals are defined as
                                  resid_response/link'(mu); see links.py for the
                                  derivatives of the link functions.

        resid_anscombe          : array
                                 Anscombe residuals; see family.py for 
                                 distribution-specific Anscombe residuals.
        
        resid_deviance          : array
                                 deviance residuals; see family.py for 
                                 distribution-specific deviance residuals.

        pearson_chi2            : float
                                  chi-Squared statistic is defined as the sum 
                                  of the squares of the Pearson residuals

        normalized_cov_params   : array
                                k*k, approximates [X.T*X]-1

    Examples
    --------
    >>> from pysal.contrib.glm.glm import GLM, GLMResults
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> self.y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> self.X = np.array(X).T
    >>> model = GLM(self.y, self.X, family=Gaussian())
    >>> results1 = model.fit()
    >>> results1.aic
    408.73548964604873
    >>> model = results1.model
    >>> params = results1.params.flatten()
    >>> predy = results1.predy
    >>> w = results1.w
    >>> results2 = GLMResults(model, params, predy, w)
    >>> results2.aic
    408.73548964604873
   
    """
    def __init__(self, model, params, mu, w):
        self.model = model
        self.n = model.n
        self.y = model.y.T.flatten()
        self.X = model.X
        self.k = model.k
        self.offset = model.offset
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.family = model.family
        self.fit_params = model.fit_params
        self.params = params
        self.w = w
        self.mu = mu.flatten()
        self._cache = {}
        self.normalized_cov_params = la.inv(spdot(self.w.T, self.w))

    @cache_readonly
    def resid_response(self):
        return (self.y-self.mu)

    @cache_readonly
    def resid_pearson(self):
        return  ((self.y-self.mu) /
                np.sqrt(self.family.variance(self.mu)))

    @cache_readonly
    def resid_working(self):
        return (self.resid_response / self.family.link.deriv(self.mu))

    @cache_readonly
    def resid_anscombe(self):
        return (self.family.resid_anscombe(self.y, self.mu))

    @cache_readonly
    def resid_deviance(self):
        return (self.family.resid_dev(self.y, self.mu))

    @cache_readonly
    def pearson_chi2(self):
        chisq = (self.y - self.mu)**2 / self.family.variance(self.mu)
        chisqsum = np.sum(chisq)
        return chisqsum

    @cache_readonly
    def null(self):
        y = np.reshape(self.y, (-1,1))
        model = self.model
        X = np.ones((len(y), 1))
        null_mod =  GLM(y, X, family=self.family, offset=self.offset, constant=False)
        return null_mod.fit().mu
   
    @cache_readonly
    def scale(self):
        if isinstance(self.family, (family.Binomial, family.Poisson)):
            return 1.
        else:
            return (((np.power(self.resid_response, 2) /
                         self.family.variance(self.mu))).sum() /
                        (self.df_resid))
    @cache_readonly
    def deviance(self):
        return self.family.deviance(self.y, self.mu)

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)
   
    @cache_readonly
    def llnull(self):
        return self.family.loglike(self.y, self.null, scale=self.scale)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y, self.mu, scale=self.scale)
    
    @cache_readonly
    def aic(self):
        if isinstance(self.family, family.QuasiPoisson):
        	return np.nan
        else:
            return -2 * self.llf + 2*(self.df_model+1)

    @cache_readonly
    def bic(self):
        return (self.deviance -
                (self.model.n - self.df_model - 1) *
                np.log(self.model.n))
    
    @cache_readonly
    def D2(self):
        return 1 - (self.deviance / self.null_deviance)

    @cache_readonly
    def adj_D2(self):
        return 1.0 - (float(self.n) - 1.0)/(float(self.n) - float(self.k)) * (1.0-self.D2)
    
    @cache_readonly
    def pseudoR2(self):
        return 1 - (self.llf/self.llnull)
    
    @cache_readonly
    def adj_pseudoR2(self):
        return 1 - ((self.llf-self.k)/self.llnull)
    
