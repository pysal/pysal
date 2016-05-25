# coding=utf-8
"""
MLE calibration for Wilson (1967) family of gravity models

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

class GravityBase(object):
    """
    Base class to set up attributes common across the family of gravity-type
    spatial interaction models

    Parameters
    ----------
    data            : pandas DataFrame
                      DataFrame containing data for model calibration
    flows           : string
                      name of column containing observed flows; depedent variable; y
    origins         : string
                      name of column containing origin unique identifiers
    destinations    : string
                      name of column containing destination unique identifiers 
    cost            : list
                      name of column containing cost variable; typically
                      distance or time
                      go from M to N in all M*N flow onservations; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'exp'
                      'exp' | 'pow'
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      filtered DataFrame if filter_intra=True
    f               : array
                      n x 1; observed flows; dependent variable; y
    o               : array of strings
                      n x 1; origin unique identifiers for n flows
    d               : array of strings
                      n x 1; destination unique identifiers for n flows
    c               : array
                      n x 1; cost associated with separation of each (o,d) pair
    cf              : function
                      cost function; used to transform cost variable
    """
    def __init__(self, data, flows, origins, destinations, cost,
            cost_func, filter_intra=True):
        if filter_intra:
        	self.dt = data[data[origins] !=
        	        data[destinations]].reset_index(level=0, drop=True)
        else:
            self.dt = data
        self.o = self.dt[origins].astype(str)
        self.d = self.dt[destinations].astype(str)
        self.f = self.dt[flows]
        self.n = self.f.shape[0]
        self.c = self.dt[cost]

        if cost_func.lower() == 'pow':
            self.cf = np.log
        elif cost_func.lower() == 'exp':
            self.cf = lambda x: x*1.0
        else:
            raise ValueError('cost_func must either be "exp" or "power"')

class Gravity(GravityBase):
    """
    Unconstrained (traditional gravity) gravity-type spatial interaction model

    Parameters
    ----------
    data            : pandas DataFrame
                      DataFrame containing data for model calibration
    flows           : string
                      name of column containing observed flows; depedent variable; y
    origins         : string
                      name of column containing origin unique identifiers
    destinations    : string
                      name of column containing destination unique identifiers 
    o_vars          : list of strings
                      names of columns containing origin attributtes
    d_vars          : list of strings
                      names of columns containing  destination attributes
    cost            : list
                      name of column containing cost variable; typically
                      distance or time
                      go from M to N in all M*N flow onservations; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'exp'
                      'exp' | 'pow'
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      filtered DataFrame if filter_intra=True
    f               : array
                      n x 1; observed flows; dependent variable; y
    o               : array of strings
                      n x 1; origin unique identifiers for n flows
    d               : array of strings
                      n x 1; destination unique identifiers for n flows
    ov              : dict{string : array}
                      keys are string of origin variable's name and values are n x 1
                      arrays of N origin variable values
    dv              : dict{string  array}
                      keys are string of destination variable's name and values
                      are n x 1 arrays of M destination variable values
    c               : array
                      n x 1; cost associated with separation of each (o,d) pair
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
    def __init__(self, data, flows, origins, destinations, o_vars, d_vars, cost,
            cost_func, filter_intra=True):
        
        GravityBase.__init__(self, data, flows, origins, destinations,
                cost, cost_func, filter_intra=True)

        self.ov = dict(zip(o_vars, [self.dt[x] for x in o_vars]))
        self.dv = dict(zip(d_vars, [self.dt[x] for x in d_vars]))
    
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
            y = self.f
            X = np.hstack((np.ones((self.n, 1)), self.ov, self.dv,
                self.cf(self.c)))
            model = sm.GLM(y, X, family = families.Poisson()).fit()
        return model

class Production(GravityBase):
    """
    Production-constrained (origin-constrained) gravity-type spatial interaction model

    Parameters
    ----------
    data            : pandas DataFrame
                      DataFrame containing data for model calibration
    flows           : string
                      name of column containing observed flows; depedent variable; y
    origins         : string
                      name of column containing origin unique identifiers
    destinations    : string
                      name of column containing destination unique identifiers 
    d_vars          : list of strings
                      names of columns containing  destination attributes
    cost            : list
                      name of column containing cost variable; typically
                      distance or time
                      go from M to N in all M*N flow onservations; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'exp'
                      'exp' | 'pow'
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      filtered DataFrame if filter_intra=True
    f               : array
                      n x 1; observed flows; dependent variable; y
    o               : array of strings
                      n x 1; origin unique identifiers for n flows
    d               : array of strings
                      n x 1; destination unique identifiers for n flows
    ov              : dict{string : array}
                      keys are string of origin variable's name and values are n x 1
                      arrays of N origin variable values
    dv              : dict{string  array}
                      keys are string of destination variable's name and values
                      are n x 1 arrays of M destination variable values
    c               : array
                      n x 1; cost associated with separation of each (o,d) pair
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
                      keys are the names of the appropriate fit statistics
                      associated with a fit framework and values are
                      correspinding statistic values

    Example
    -------
    TODO

    """
    def __init__(self, data, flows, origins, destinations, d_vars, cost,
            cost_func, filter_intra=True):
        
        GravityBase.__init__(self, data, flows, origins, destinations,
                cost, cost_func, filter_intra=True)
        
        self.ov = {}
        self.dv = dict(zip(d_vars, [self.dt[x] for x in d_vars]))

class Attraction(GravityBase):
    """
    Attraction-constrained (destination-constrained) gravity-type spatial interaction model

    Parameters
    ----------
    data            : pandas DataFrame
                      DataFrame containing data for model calibration
    flows           : string
                      name of column containing observed flows; depedent variable; y
    origins         : string
                      name of column containing origin unique identifiers
    destinations    : string
                      name of column containing destination unique identifiers 
    o_vars          : list of strings
                      names of columns containing origin attributtes
    cost            : list
                      name of column containing cost variable; typically
                      distance or time
                      go from M to N in all M*N flow onservations; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'exp'
                      'exp' | 'pow'
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      filtered DataFrame if filter_intra=True
    f               : array
                      n x 1; observed flows; dependent variable; y
    o               : array of strings
                      n x 1; origin unique identifiers for n flows
    d               : array of strings
                      n x 1; destination unique identifiers for n flows
    ov              : dict{string : array}
                      keys are string of origin variable's name and values are n x 1
                      arrays of N origin variable values
    dv              : dict{string  array}
                      keys are string of destination variable's name and values
                      are n x 1 arrays of M destination variable values
    c               : array
                      n x 1; cost associated with separation of each (o,d) pair
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
                      keys are the names of the appropriate fit statistics
                      associated with a fit framework and values are
                      correspinding statistic values

    Example
    -------
    TODO

    """
    def __init__(self, data, flows, origins, destinations, o_vars, cost,
            cost_func, filter_intra=True):
        
        GravityBase.__init__(self, data, flows, origins, destinations,
                 cost, cost_func, filter_intra=True)

        self.ov = dict(zip(o_vars, [self.dt[x] for x in o_vars]))
        self.dv = {}

class Doubly(GravityBase):
    """
    Doubly-constrained gravity-type spatial interaction model

    Parameters
    ----------
    data            : pandas DataFrame
                      DataFrame containing data for model calibration
    flows           : string
                      name of column containing observed flows; depedent variable; y
    origins         : string
                      name of column containing origin unique identifiers
    destinations    : string
                      name of column containing destination unique identifiers 
    cost            : list
                      name of column containing cost variable; typically
                      distance or time
                      go from M to N in all M*N flow onservations; typically distance or time
    cost_func       : string
                      functional form of the cost function; default is 'exp'
                      'exp' | 'pow'
    filter_intra    : boolean
                      True (default) to filter intra-zonal flows

    Attributes
    ----------
    dt              : pandas DataFrame
                      filtered DataFrame if filter_intra=True
    f               : array
                      n x 1; observed flows; dependent variable; y
    o               : array of strings
                      n x 1; origin unique identifiers for n flows
    d               : array of strings
                      n x 1; destination unique identifiers for n flows
    ov              : dict{string : array}
                      keys are string of origin variable's name and values are n x 1
                      arrays of N origin variable values
    dv              : dict{string  array}
                      keys are string of destination variable's name and values
                      are n x 1 arrays of M destination variable values
    c               : array
                      n x 1; cost associated with separation of each (o,d) pair
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
                      keys are the names of the appropriate fit statistics
                      associated with a fit framework and values are
                      correspinding statistic values

    Example
    -------
    TODO

    """
    def __init__(self, data, flows, origins, destinations, cost,
            cost_func, filter_intra=True):
        
        GravityBase.__init__(self, data, flows, origins,
                destinations, cost, cost_func, filter_intra=True)

        self.ov = {}
        self.dv = {}

