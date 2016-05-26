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
import statsmodels.api as sm
from statsmodels.api import families 
from statsmodels.tools.tools import categorical
from pysal.spreg import user_output as User
from count_base import CountModel

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
    params          : array
                      estimated parameters
    se              : array
                      standard errors associated with estimated parameters
    t_stats         : array
                      t-statistics associated with estimated parameters for
                      hypothesis testing
    fitted          : array
                      n x 1; flow values produced by calibrated model
    fit_stats       : dict{"statistic name": statistic value}
    
    Example
    -------
    TODO

    """
    def __init__(self, flows, cost, cost_func='pow', o_vars=None, d_vars=None,
            origins=None, destinations=None, constant=True):
        flows = np.reshape(flows, (-1,1))
        cost = np.reshape(cost, (-1,1))
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
        X = np.empty((self.n, 0))
        if constant:
            X = User.check_constant(X)

        if isinstance(self, Production) | isinstance(self, Doubly):
            o_dummies = categorical(origins.astype(str), drop=True)
            X = np.hstack((X, o_dummies))
        if isinstance(self, Attraction) | isinstance(self, Doubly):
            d_dummies = categorical(destinations.astype(str), drop=True)
            X = np.hstack((X, d_dummies))

        if self.ov is not None:
            X = np.hstack((X, np.log(np.reshape(self.ov, (-1,1)))))
        if self.dv is not None:
            X = np.hstack((X, np.log(np.reshape(self.dv, (-1,1)))))
        X = np.hstack((X, self.cf(np.reshape(self.c, (-1,1)))))

        CountModel.__init__(self, y, X)
        self.fit()
        

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
                      estimated parameters
    se              : array
                      standard errors associated with estimated parameters
    t_stats         : array
                      t-statistics associated with estimated parameters for
                      hypothesis testing
    fitted          : array
                      n x 1; flow values produced by calibrated model
    fit_stats       : dict{"statistic name": statistic value}
    
    Example
    -------
    TODO

    """
    def __init__(self, flows, o_vars, d_vars, cost,
            cost_func):
        User.check_arrays(flows, o_vars, d_vard, cost)
        
        BaseGravity.__init__(self, flows, cost,
                cost_func, o_vars=o_vars, d_vars=d_vars)

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
                      estimated parameters
    se              : array
                      standard errors associated with estimated parameters
    t_stats         : array
                      t-statistics associated with estimated parameters for
                      hypothesis testing
    fitted          : array
                      n x 1; flow values produced by calibrated model
    fit_stats       : dict{"statistic name": statistic value}

    Example
    -------
    TODO

    """
    def __init__(self, flows, origins, d_vars, cost, cost_func):
        User.check_arrays(flows, origins, d_vars, cost
                )
        BaseGravity.__init__(self, flows, cost, cost_func, d_vars=d_vars,
                origins=origins)
        
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
                      estimated parameters
    se              : array
                      standard errors associated with estimated parameters
    t_stats         : array
                      t-statistics associated with estimated parameters for
                      hypothesis testing
    fitted          : array
                      n x 1; flow values produced by calibrated model
    fit_stats       : dict{"statistic name": statistic value}

    Example
    -------
    TODO

    """
    def __init__(self, flows, destinations, o_vars, cost, cost_func):
        User.check_arrays(flows, destinations, o_vars, cost)

        BaseGravity.__init__(self, flows, cost, cost_func, o_vars=o_vars,
                 destinations=destinations)

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
                      estimated parameters
    se              : array
                      standard errors associated with estimated parameters
    t_stats         : array
                      t-statistics associated with estimated parameters for
                      hypothesis testing
    fitted          : array
                      n x 1; flow values produced by calibrated model
    fit_stats       : dict{"statistic name": statistic value}

    Example
    -------
    TODO

    """
    def __init__(self, flows, origins, destinations, cost, cost_func):
        User.check_arrays(flows, origins, destinations, cost)
        #maybe a check for equal number of origins and destinations

        BaseGravity.__init__(self, flows, cost, cost_func, origins=origins, 
                destinations=destinations)

