"""
Base classes for count models
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
import statsmodels.api as sm
from statsmodels.api import families

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
                  
    Attributes
    ----------
    y           : array
                  n x 1; n observations of the depedent variable
    X           : array
                  n x k; design matrix of k explanatory variables

    """
    def __init__(self, y, X):
        self.y = self._check_counts(y)
        self.X = X
        
    def _check_counts(self, y):
        if (y.dtype == 'int64') | (y.dtype == 'int32'):
        	return y
        else:
        	raise TypeError('Dependent variable (y) must be composed of integers')

    def fit(self, framework='GLM', method='iwls'):
        '''
        estimates parameters (coefficients) of spatial interaction model
            
        Parameters
        ----------
        framework           : string
                            estimation framework; default is GLM
                            "GLM" | "entropy"
        method              : string
                            estimation method for GLM framework; default is
                            itertaively weighted least sqaures (iwls)
                            "iwls" | "TODO - add other methods"

       Returns
       -------
       betas                : array
                              K+L+D+1 x 1; estimated parameters for
                              origin/destination/cost variables and constant
        '''
        if (framework.lower() == 'glm'):
            
            results = sm.GLM(self.y, self.X, family = families.Poisson()).fit() 
            self.params = results.params
            self.se = results.bse
            self.tvals = results.tvalues
            self.fitted = results.fittedvalues
            self.fit_stats = {}
            self.fit_stats['aic'] = results.aic  
