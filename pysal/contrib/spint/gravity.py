# coding=utf-8
"""
 Wilsonian (1967) family of gravity-type spatial interaction models

References
----------

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations
 and Applications. London: Kluwer Academic Publishers.

Wilson, A. G. (1967). A statistical theory of spatial distribution models.
 Transportation Research, 1, 253–269.

"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
from scipy import sparse as sp
import statsmodels.api as sm
from statsmodels.api import families 
from statsmodels.tools.tools import categorical
from sparse_categorical import spcategorical
from pysal.spreg import user_output as User
from count_model import CountModel

class BaseGravity(CountModel):
    """
    Base class to set up attributes common across the family of gravity-type
    spatial interaction models

    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    origins         : array of strings
                      n x 1; unique identifiers of origins of n flows
    destinations    : array of strings
                      n x 1; unique identifiers of destinations of n flows 
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'pow'
                      'exp' | 'pow'
    o_vars          : array (optional)
                      n x k; k attributes for each origin of  n flows; default
                      is None
    d_vars          : array (optional)
                      n x k; k attributes for each destination of n flows;
                      default is None

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    ov              : array 
                      n x k; k attributes for each origin of n flows
    dv              : array
                      n x k; k attributes for each destination of n flows
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
    results       : object
                    Full results from estimated model. May contain addtional
                    diagnostics
    Example
    -------
    TODO

    """
    def __init__(self, flows, cost, cost_func='pow', o_vars=None, d_vars=None,
            origins=None, destinations=None, constant=False, framework='GLM',
            SF=None, CD=None, Lag=None):
        n = User.check_arrays(flows, cost)
        User.check_y(flows, n)
        self.n = n
        self.f = flows
        self.c = cost
        self.ov = o_vars
        self.dv = d_vars

        if cost_func.lower() == 'pow':
            self.cf = np.log
        elif cost_func.lower() == 'exp':
            self.cf = lambda x: x*1.0
        else:
            raise ValueError('cost_func must either be "exp" or "power"')

        y = np.reshape(self.f, (-1,1))
        if isinstance(self,  Gravity):
            X = np.empty((self.n, 0))
        else:
            X = sp.csr_matrix((self.n, 1))
        if isinstance(self, Attraction) | isinstance(self, Doubly):
            d_dummies = spcategorical(destinations.flatten().astype(str))
            X = sp.hstack((X, d_dummies))
        if isinstance(self, Production) | isinstance(self, Doubly):
            o_dummies = spcategorical(origins.flatten().astype(str)) 
            X = sp.hstack((X, o_dummies))
        if isinstance(self, Doubly):
            X = sp.csr_matrix(X)
            X = X[:,1:]
        if self.ov is not None:	
            if isinstance(self, Gravity):
                X = np.hstack((X, np.log(np.reshape(self.ov, (-1,1)))))
            else:
                ov = sp.csr_matrix(np.log(np.reshape(self.ov, ((-1,1)))))
                X = sp.hstack((X, ov))
        if self.dv is not None:    	
            if isinstance(self, Gravity):
                X = np.hstack((X, np.log(np.reshape(self.dv, (-1,1)))))
            else:
                dv = sp.csr_matrix(np.log(np.reshape(self.dv, ((-1,1)))))
                X = sp.hstack((X, dv))
        if isinstance(self, Gravity):
            X = np.hstack((X, self.cf(np.reshape(self.c, (-1,1)))))
        else:
            c = sp.csr_matrix(self.cf(np.reshape(self.c, (-1,1))))
            X = sp.hstack((X, c))
            X = sp.csr_matrix(X)
            X = X[:,1:]
        if not isinstance(self, (Gravity, Production, Attraction, Doubly)):
            X = self.cf(np.reshape(self.c, (-1,1)))

        if SF:
        	raise NotImplementedError('Spatial filter model not yet implemented')
        if CD:
        	raise NotImplementedError('Competing destination model not yet implemented')
        if Lag:
        	raise NotImplementedError('Spatial Lag autoregressive model not yet implemented')
        
        CountModel.__init__(self, y, X, constant=constant)
        if (framework.lower() == 'glm'):
            results = self.fit(framework='glm')
        else:
            raise NotImplementedError('Only GLM is currently implemented')

        self.params = results.params
        self.yhat = results.yhat
        self.cov_params = results.cov_params
        self.std_err = results.std_err
        self.pvalues = results.pvalues
        self.tvalues = results.tvalues
        self.deviance = results.deviance
        self.llf = results.llf
        self.aic = results.aic
        self.full_results = results
            
class Gravity(BaseGravity):
    """
    Unconstrained (traditional gravity) gravity-type spatial interaction model

    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'pow'
                      'exp' | 'pow'
    o_vars          : array (optional)
                      n x k; k attributes for each origin of  n flows; default
                      is None
    d_vars          : array (optional)
                      n x k; k attributes for each destination of n flows;
                      default is None

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    ov              : array 
                      n x k; k attributes for each origin of n flows
    dv              : array 
                      n x k; k attributes for each destination of n flows
    params          : array
                      n*k, estimared beta coefficients
    yhat            : array
                      n*1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k*1, standard errors of betas
    pvalues         : array
                      k*1, two-tailed pvalues of parameters
    tvalues         : array
                      k*1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    aic             : float 
                      Akaike information criterion
    resid           : array
                      response residuals; defined as y-yhat
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    TODO

    """
    def __init__(self, flows, o_vars, d_vars, cost,
            cost_func, constant=False, framework='GLM', SF=None, CD=None,
            Lag=None):
        flows = np.reshape(flows, (-1,1))
        o_vars = np.reshape(o_vars, (-1,1))
        d_vars = np.reshape(d_vars, (-1,1))
        cost = np.reshape(cost, (-1,1))
        User.check_arrays(flows, o_vars, d_vars, cost)
        
        BaseGravity.__init__(self, flows, cost,
                cost_func, o_vars=o_vars, d_vars=d_vars, constant=constant,
                framework=framework, SF=SF, CD=CD, Lag=Lag)
        

class Production(BaseGravity):
    """
    Production-constrained (origin-constrained) gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    origins         : array of strings
                      n x 1; unique identifiers of origins of n flows
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'pow'
                      'exp' | 'pow'
    d_vars          : array (optional)
                      n x k; k attributes for each destination of n flows;
                      default is None

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    dv              : array 
                      n x k; k attributes for each destination of n flows
    params          : array
                      n*k, estimared beta coefficients
    yhat            : array
                      n*1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k*1, standard errors of betas
    pvalues         : array
                      k*1, two-tailed pvalues of parameters
    tvalues         : array
                      k*1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    aic             : float 
                      Akaike information criterion
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    TODO

    """
    def __init__(self, flows, origins, d_vars, cost, cost_func, constant=False,
            framework='GLM', SF=None, CD=None, Lag=None):
        flows = np.reshape(flows, (-1,1))
        origins = np.reshape(origins, (-1,1))
        d_vars = np.reshape(d_vars, (-1,1))
        cost = np.reshape(cost, (-1,1))
        User.check_arrays(flows, origins, d_vars, cost)
       
        BaseGravity.__init__(self, flows, cost, cost_func, d_vars=d_vars,
                origins=origins, constant=constant, framework=framework,
                SF=SF, CD=CD, Lag=Lag)
        
class Attraction(BaseGravity):
    """
    Attraction-constrained (destination-constrained) gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    destinations    : array of strings
                      n x 1; unique identifiers of destinations of n flows 
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'pow'
                      'exp' | 'pow'
    o_vars          : array (optional)
                      n x k; k attributes for each origin of  n flows; default
                      is None

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    ov              : array
                      n x k; k attributes for each origin of n flows
    params          : array
                      n*k, estimared beta coefficients
    yhat            : array
                      n*1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k*1, standard errors of betas
    pvalues         : array
                      k*1, two-tailed pvalues of parameters
    tvalues         : array
                      k*1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    aic             : float 
                      Akaike information criterion
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    TODO

    """
    def __init__(self, flows, destinations, o_vars, cost, cost_func,
            constant=False, framework='GLM', SF=None, CD=None, Lag=None):
        flows = np.reshape(flows, (-1,1))
        o_vars = np.reshape(o_vars, (-1,1))
        destinations = np.reshape(destinations, (-1,1))
        cost = np.reshape(cost, (-1,1))
        User.check_arrays(flows, destinations, o_vars, cost)

        BaseGravity.__init__(self, flows, cost, cost_func, o_vars=o_vars,
                 destinations=destinations, constant=constant,
                 framework=framework, SF=SF, CD=CD, Lag=Lag)

class Doubly(BaseGravity):
    """
    Doubly-constrained gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    origins         : array of strings
                      n x 1; unique identifiers of origins of n flows
    destinations    : array of strings
                      n x 1; unique identifiers of destinations of n flows 
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'pow'
                      'exp' | 'pow'

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    params          : array
                      n*k, estimared beta coefficients
    yhat            : array
                      n*1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k*1, standard errors of betas
    pvalues         : array
                      k*1, two-tailed pvalues of parameters
    tvalues         : array
                      k*1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    aic             : float 
                      Akaike information criterion
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    TODO

    """
    def __init__(self, flows, origins, destinations, cost, cost_func,
            constant=False, framework='GLM', SF=None, CD=None, Lag=None):

        flows = np.reshape(flows, (-1,1))
        origins = np.reshape(origins, (-1,1))
        destinations = np.reshape(destinations, (-1,1))
        cost = np.reshape(cost, (-1,1))
        User.check_arrays(flows, origins, destinations, cost)

        BaseGravity.__init__(self, flows, cost, cost_func, origins=origins, 
                destinations=destinations, constant=constant,
                framework=framework, SF=SF, CD=CD, Lag=Lag)

