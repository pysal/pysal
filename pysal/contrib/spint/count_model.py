"""
Base clases for count models
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
import sys
#from pysal.contrib.glm.glm import GLM
#from pysal.contrib.glm.family import Poisson
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/glm')
from glm import GLM
from family import Poisson

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

    def fit(self, framework='GLM'):
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
            results = GLM(self.y, self.X, family = Poisson(), constant=self.constant).fit()
            return CountModelResults(results, framework=framework)
   
        else:
            raise NotImplemented('Poisson GLM is the only count model currently implemented')

class CountModelResults(object):
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
        aic           : float 
                        Akaike information criterion
        resid         : array
                        response residuals; defined as y-mu
        

    """
    def __init__(self, results, framework='GLM'):
        self.y = results.y
        self.X = results.X
        self.family = results.family
        self.params = results.params
        self.aic = results.aic
        self.df_model = results.df_model
        self.df_resid = results.df_resid
        self.llf = results.llf
        self.yhat = results.mu
        self.deviance = results.deviance
        self.n = results.n
        self.k = results.k
        self.resid = results.resid_response
        self.cov_params = results.cov_params()
        self.std_err = results.bse
        self.pvalues = results.pvalues
        self.tvalues = results.tvalues
        self.model = results

    
