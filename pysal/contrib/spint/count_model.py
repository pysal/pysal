"""
CountModel class for dispatching different types of count models and different
types of estimation technqiues.
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
from pysal.contrib.glm.glm import GLM
from pysal.contrib.glm.family import Poisson, QuasiPoisson

class CountModel(object):
    """
    Base class for variety of count-based models such as Poisson, negative binomial,
    etc. of the exponetial family. 

    Parameters
    ----------
    y           : array
                  n x 1; n observations of the depedent variable
    X           : array
                  n x k; design matrix of k explanatory variables
    family      : instance of class 'family'
                  default is Poisson()
    constant    : boolean
                  True if intercept should be estimated and false otherwise.
                  Default is True.

                  
    Attributes
    ----------
    y           : array
                  n x 1; n observations of the depedent variable
    X           : array
                  n x k; design matrix of k explanatory variables
    fitted      : boolean
                  False is model has not been fitted and True if it has been
                  successfully fitted. Deault is False. 
    constant    : boolean
                  True if intercept should be estimated and false otherwise.
                  Default is True.

    Example
    -------
    >>> import pysal.contrib.spint.count_model import CountModel
    >>> from pysal.
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y =  np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> self.y = np.round(y).astype(int)
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> self.X = np.array(X).T
    >>> model = CountModel(self.y, self.X, family=Poisson())
    >>> results = model.fit('GLM')
    >>> results.params
    array([3.92159085, 0.01183491, -0.01371397])

    """
    def __init__(self, y, X, family = Poisson(), constant = True):
        self.y = self._check_counts(y)
        self.X = X
        self.constant = constant
    def _check_counts(self, y):
        if (y.dtype == 'int64') | (y.dtype == 'int32'):
        	return y
        else:
        	raise TypeError('Dependent variable (y) must be composed of integers')

    def fit(self, framework='GLM', Quasi=False):
        """
        Method that fits a particular count model usign the appropriate
        estimation technique. Models include Poisson GLM, Negative Binomial GLM,
        Quasi-Poisson - at the moment Poisson GLM is the only option.

        TODO: add zero inflated variants and hurdle variants.

        Parameters
        ----------
        framework           : string
                            estimation framework; default is GLM
                             "GLM" | "QUASI" | 
        """
        if (framework.lower() == 'glm'):
            if not Quasi:
                results = GLM(self.y, self.X, family = Poisson(), constant=self.constant).fit()
            else:
                results = GLM(self.y, self.X, family = QuasiPoisson(), constant=self.constant).fit()
            return CountModelResults(results)
   
        else:
            raise NotImplemented('Poisson GLM is the only count model currently implemented')

class CountModelResults(object):
    """
    Results of estimated GLM and diagnostics.

    Parameters
    ----------
        results       : GLM object
                        Pointer to GLMResults object with estimated parameters
                        and diagnostics.
    
    Attributes
    ----------
        model         : GLM Object
                        Points to model object for which parameters have been
                        estimated. May contain additional diagnostics.
        y             : array
                        n*1, dependent variable.
        X             : array
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
                        routine.
        params        : array
                        n*k, estimared beta coefficients
        yhat          : array
                        n*1, predicted value of y (i.e., fittedvalues)
        cov_params    : array
                        Variance covariance matrix (kxk) of betas
        std_err       : array
                        k*1, standard errors of betas
        pvalues       : array
                        k*1, two-tailed pvalues of parameters
        tvalues       : array
                        k*1, the tvalues of the standard errors
        deviance      : float
                        value of the deviance function evalued at params;
                        see family.py for distribution-specific deviance
        llf           : float
                        value of the loglikelihood function evalued at params;
                        see family.py for distribution-specific loglikelihoods
        llnull        : float
                        value of the loglikelihood function evaluated with only an
                        intercept; see family.py for distribution-specific
                        loglikelihoods
        AIC           : float 
                        Akaike information criterion
        resid         : array
                        response residuals; defined as y-mu
        
        resid_dev     : array
                        k x 1, residual deviance of model
        D2            : float
                        percentage of explained deviance
        adj_D2        : float

        pseudo_R2       : float
                        McFadden's pseudo R2  (coefficient of determination) 
        adj_pseudoR2    : float
                        adjusted McFadden's pseudo R2

    """
    def __init__(self, results):
        self.y = results.y
        self.X = results.X
        self.family = results.family
        self.params = results.params
        self.AIC = results.aic
        self.df_model = results.df_model
        self.df_resid = results.df_resid
        self.llf = results.llf
        self.llnull = results.llnull
        self.yhat = results.mu
        self.deviance = results.deviance
        self.n = results.n
        self.k = results.k
        self.resid = results.resid_response
        self.resid_dev = results.resid_deviance
        self.cov_params = results.cov_params()
        self.std_err = results.bse
        self.pvalues = results.pvalues
        self.tvalues = results.tvalues
        self.D2 = results.D2
        self.adj_D2 = results.adj_D2
        self.pseudoR2 = results.pseudoR2
        self.adj_pseudoR2 = results.adj_pseudoR2
        self.model = results

    
