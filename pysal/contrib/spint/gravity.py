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

from types import FunctionType
import numpy as np
from scipy import sparse as sp
from pysal.spreg import user_output as User
from pysal.spreg.utils import sphstack
from pysal.contrib.glm.utils import cache_readonly
from count_model import CountModel
from utils import sorensen, srmse, spcategorical

class BaseGravity(CountModel):
    """
    Base class to set up gravity-type spatial interaction models and dispatch
    estimaton technqiues.

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
    cost_func       : string or function that has scalar input and output
                      functional form of the cost function;
                      'exp' | 'pow' | custom function
    o_vars          : array (optional)
                      n x p; p attributes for each origin of  n flows; default
                      is None
    d_vars          : array (optional)
                      n x p; p attributes for each destination of n flows;
                      default is None
    constant        : boolean
                      True to include intercept in model; false by default
    framework       : string
                      estimation technique; currently only 'GLM' is avaialble
    Quasi           : boolean
                      True to estimate QuasiPoisson model; should result in same
                      parameters as Poisson but with altered covariance; default
                      to true which estimates Poisson model
    SF              : array
                      n x 1; eigenvector spatial filter to include in the model;
                      default to None which does not include a filter; not yet
                      implemented
    CD              : array
                      n x 1; competing destination term that accounts for the
                      likelihood that alternative destinations are considered
                      along with each destination under consideration for every
                      OD pair; defaults to None which does not include a CD
                      term; not yet implemented
    Lag             : W object
                      spatial weight for n observations (OD pairs) used to
                      construct a spatial autoregressive model and estimator;
                      defaults to None which does not include an autoregressive
                      term; not yet implemented


    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    k               : integer
                      number of parameters
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    ov              : array 
                      n x p(o); p attributes for each origin of n flows
    dv              : array
                      n x p(d); p attributes for each destination of n flows
    constant        : boolean
                      True to include intercept in model; false by default
    y               : array
                      n x 1; dependent variable used in estimation including any
                      transformations
    X               : array
                      n x k, design matrix used in estimation
    params          : array
                      n x k, k estimated beta coefficients; k = p(o) + p(d) + 1
    yhat            : array
                      n x 1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (k x k) of betas
    std_err         : array
                      k x 1, standard errors of betas
    pvalues         : array
                      k x 1, two-tailed pvalues of parameters
    tvalues         : array
                      k x 1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    resid_dev       : array
                      n x 1, residual deviance of model
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    llnull          : float
                      value of the loglikelihood function evaluated with only an
                      intercept; see family.py for distribution-specific
                      loglikelihoods
    aic             : float
                      Akaike information criterion
    D2              : float
                      percentage of explained deviance
    adj_D2          : float
                      adjusted percentage of explained deviance
    pseudo_R2       : float
                      McFadden's pseudo R2  (coefficient of determination) 
    adj_pseudoR2    : float
                      adjusted McFadden's pseudo R2
    SRMSE           : float
                      standardized root mean square error
    SSI             : float
                      Sorensen similarity index
    results         : object
                      full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    >>> import numpy as np
    >>> import pysal
    >>> from pysal.contrib.spint.gravity import BaseGravity
    >>> db = pysal.open(pysal.examples.get_path('nyc_bikes_ct.csv'))
    >>> cost = np.array(db.by_col('tripduration')).reshape((-1,1))
    >>> flows = np.array(db.by_col('count')).reshape((-1,1))
    >>> model = BaseGravity(flows, cost)
    >>> model.params
    array([ 0.92860101])

    """
    def __init__(self, flows, cost, cost_func='pow', o_vars=None, d_vars=None,
            origins=None, destinations=None, constant=False, framework='GLM',
            SF=None, CD=None, Lag=None, Quasi=False):
        n = User.check_arrays(flows, cost)
        #User.check_y(flows, n)
        self.n = n
        self.f = flows
        self.c = cost
        self.ov = o_vars
        self.dv = d_vars
        if type(cost_func) == str:
            if cost_func.lower() == 'pow':
                self.cf = np.log
            elif cost_func.lower() == 'exp':
                self.cf = lambda x: x*1.0
        elif (type(cost_func) == FunctionType) | (type(cost_func) == np.ufunc):
            self.cf = cost_func
        else:
            raise ValueError("cost_func must be 'exp', 'pow' or a valid"
            " function that has a scalar as a input and output")

        y = np.reshape(self.f, (-1,1))
        if isinstance(self, Gravity):
            X = np.empty((self.n, 0))
        else:
            X = sp.csr_matrix((self.n, 1))
        if isinstance(self, Attraction) | isinstance(self, Doubly):
            d_dummies = spcategorical(destinations.flatten())
            X = sphstack(X, d_dummies, array_out=False)
        if isinstance(self, Production) | isinstance(self, Doubly):
            o_dummies = spcategorical(origins.flatten())
            X = sphstack(X, o_dummies, array_out=False)
        if isinstance(self, Doubly):
            X = X[:,1:]
        if self.ov is not None:	
            if isinstance(self, Gravity):
                for each in range(self.ov.shape[1]):
                    X = np.hstack((X, np.log(np.reshape(self.ov[:,each], (-1,1)))))
            else:
                for each in range(self.ov.shape[1]):
                    ov = sp.csr_matrix(np.log(np.reshape(self.ov[:,each], ((-1,1)))))
                    X = sphstack(X, ov, array_out=False)
        if self.dv is not None:    	
            if isinstance(self, Gravity):
                for each in range(self.dv.shape[1]):
                    X = np.hstack((X, np.log(np.reshape(self.dv[:,each], (-1,1)))))
            else:
                for each in range(self.dv.shape[1]):
                    dv = sp.csr_matrix(np.log(np.reshape(self.dv[:,each], ((-1,1)))))
                    X = sphstack(X, dv, array_out=False)
        if isinstance(self, Gravity):
            X = np.hstack((X, self.cf(np.reshape(self.c, (-1,1)))))
        else:
            c = sp.csr_matrix(self.cf(np.reshape(self.c, (-1,1))))
            X = sphstack(X, c, array_out=False)
            X = X[:,1:]#because empty array instantiated with extra column
        if not isinstance(self, (Gravity, Production, Attraction, Doubly)):
            X = self.cf(np.reshape(self.c, (-1,1)))
        if SF:
        	raise NotImplementedError("Spatial filter model not yet implemented")
        if CD:
        	raise NotImplementedError("Competing destination model not yet implemented")
        if Lag:
        	raise NotImplementedError("Spatial Lag autoregressive model not yet implemented")
        
        CountModel.__init__(self, y, X, constant=constant)
        if (framework.lower() == 'glm'):
            if not Quasi:
                results = self.fit(framework='glm')
            else:
                results = self.fit(framework='glm', Quasi=True)
        else:
            raise NotImplementedError('Only GLM is currently implemented')

        self.params = results.params
        self.yhat = results.yhat
        self.cov_params = results.cov_params
        self.std_err = results.std_err
        self.pvalues = results.pvalues
        self.tvalues = results.tvalues
        self.deviance = results.deviance
        self.resid_dev = results.resid_dev
        self.llf = results.llf
        self.llnull = results.llnull
        self.aic = results.aic
        self.k = results.k
        self.D2 = results.D2
        self.adj_D2 = results.adj_D2
        self.pseudoR2 = results.pseudoR2
        self.adj_pseudoR2 = results.adj_pseudoR2
        self.results = results
        self._cache = {}

    @cache_readonly
    def SSI(self):
        return sorensen(self)

    @cache_readonly
    def SRMSE(self):
        return srmse(self)

    def reshape(self, array):
        if type(array) == np.ndarray:
            return array.reshape((-1,1))
        elif type(array) == list:
            return np.array(array).reshape((-1,1))
        else:
            raise TypeError("input must be an numpy array or list that can be coerced"
                    " into the dimensions n x 1")
    
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
    cost_func       : string or function that has scalar input and output
                      functional form of the cost function;
                      'exp' | 'pow' | custom function
    o_vars          : array (optional)
                      n x p; p attributes for each origin of  n flows; default
                      is None
    d_vars          : array (optional)
                      n x p; p attributes for each destination of n flows;
                      default is None
    constant        : boolean
                      True to include intercept in model; false by default
    framework       : string
                      estimation technique; currently only 'GLM' is avaialble
    Quasi           : boolean
                      True to estimate QuasiPoisson model; should result in same
                      parameters as Poisson but with altered covariance; default
                      to true which estimates Poisson model
    SF              : array
                      n x 1; eigenvector spatial filter to include in the model;
                      default to None which does not include a filter; not yet
                      implemented
    CD              : array
                      n x 1; competing destination term that accounts for the
                      likelihood that alternative destinations are considered
                      along with each destination under consideration for every
                      OD pair; defaults to None which does not include a CD
                      term; not yet implemented
    Lag             : W object
                      spatial weight for n observations (OD pairs) used to
                      construct a spatial autoregressive model and estimator;
                      defaults to None which does not include an autoregressive
                      term; not yet implemented

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    k               : integer
                      number of parameters
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    ov              : array 
                      n x p(o); p attributes for each origin of n flows
    dv              : array 
                      n x p(d); p attributes for each destination of n flows
    constant        : boolean
                      True to include intercept in model; false by default
    y               : array
                      n x 1; dependent variable used in estimation including any
                      transformations
    X               : array
                      n x k, design matrix used in estimation
    params          : array
                      n x k, k estimated beta coefficients; k = p(o) + p(d) + 1
    yhat            : array
                      n x 1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k x 1, standard errors of betas
    pvalues         : array
                      k x 1, two-tailed pvalues of parameters
    tvalues         : array
                      k x 1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    resid_dev       : array
                      n x 1, residual deviance of model
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    llnull          : float
                      value of the loglikelihood function evaluated with only an
                      intercept; see family.py for distribution-specific
                      loglikelihoods
    aic             : float 
                      Akaike information criterion
    D2              : float
                      percentage of explained deviance
    adj_D2          : float
                      adjusted percentage of explained deviance
    pseudo_R2       : float
                      McFadden's pseudo R2  (coefficient of determination) 
    adj_pseudoR2    : float
                      adjusted McFadden's pseudo R2
    SRMSE           : float
                      standardized root mean square error
    SSI             : float
                      Sorensen similarity index
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    >>> import numpy as np
    >>> import pysal
    >>> from pysal.contrib.spint.gravity import Gravity
    >>> db = pysal.open(pysal.examples.get_path('nyc_bikes_ct.csv'))
    >>> cost = np.array(db.by_col('tripduration')).reshape((-1,1))
    >>> flows = np.array(db.by_col('count')).reshape((-1,1))
    >>> o_cap = np.array(db.by_col('o_cap')).reshape((-1,1))
    >>> d_cap = np.array(db.by_col('d_cap')).reshape((-1,1))
    >>> model = Gravity(flows, o_cap, d_cap, cost, 'exp')
    >>> model.params
    array([ 0.87911778,  0.71080687, -0.00194626])
    
    """
    def __init__(self, flows, o_vars, d_vars, cost,
            cost_func, constant=False, framework='GLM', SF=None, CD=None,
            Lag=None, Quasi=False):
        self.f = np.reshape(flows, (-1,1))
        if len(o_vars.shape) > 1:
            p = o_vars.shape[1]
        else:
            p = 1
        self.ov = np.reshape(o_vars, (-1,p))
        if len(d_vars.shape) > 1:
            p = d_vars.shape[1]
        else:
            p = 1
        self.dv = np.reshape(d_vars, (-1,p))
        self.c = np.reshape(cost, (-1,1))
        #User.check_arrays(self.f, self.ov, self.dv, self.c)
        
        BaseGravity.__init__(self, self.f, self.c,
                cost_func=cost_func, o_vars=self.ov, d_vars=self.dv, constant=constant,
                framework=framework, SF=SF, CD=CD, Lag=Lag, Quasi=Quasi)
        
    def local(self, loc_index, locs):
        """
        Calibrate local models for subsets of data from a single location to all
        other locations
        
        Parameters
        ----------
        loc_index   : n x 1 array of either origin or destination id label for
                      flows; must be explicitly provided for local version of
                      basic gravity model since these are not passed to the
                      global model. 
                    
        locs        : iterable of either origin or destination labels for which
                      to calibrate local models; must also be explicitly
                      provided since local gravity models can be calibrated from origins
                      or destinations. If all origins are also destinations and
                      a local model is desired for each location then use
                      np.unique(loc_index)

        Returns
        -------
        results     : dict where keys are names of model outputs and diagnostics
                      and values are lists of location specific values. 
        """
        results = {}
        covs = self.ov.shape[1] + self.dv.shape[1] + 1
        results['aic'] = []
        results['deviance'] = []
        results['pseudoR2'] = []
        results['adj_pseudoR2'] = []
        results['D2'] = []
        results['adj_D2'] = []
        results['SSI'] = []
        results['SRMSE'] = []
        for cov in range(covs):
            results['param' + str(cov)] = []
            results['pvalue' + str(cov)] = []
            results['tvalue' + str(cov)] = []
        for loc in locs:
            subset = loc_index == loc
            f = self.reshape(self.f[subset])
            o_vars = self.ov[subset.reshape(self.ov.shape[0]),:]
            d_vars = self.dv[subset.reshape(self.dv.shape[0]),:]
            dij = self.reshape(self.c[subset])
            model = Gravity(f, o_vars, d_vars, dij, self.cf)
            results['aic'].append(model.aic)
            results['deviance'].append(model.deviance)
            results['pseudoR2'].append(model.pseudoR2)
            results['adj_pseudoR2'].append(model.adj_pseudoR2)
            results['D2'].append(model.D2)
            results['adj_D2'].append(model.adj_D2)
            results['SSI'].append(model.SSI)
            results['SRMSE'].append(model.SRMSE)
            for cov in range(covs):
                results['param' + str(cov)].append(model.params[cov])
                results['pvalue' + str(cov)].append(model.pvalues[cov])
                results['tvalue' + str(cov)].append(model.tvalues[cov])
        return results

class Production(BaseGravity):
    """
    Production-constrained (origin-constrained) gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    origins         : array of strings
                      n x 1; unique identifiers of origins of n flows; when
                      there are many origins it will be faster to use integers
                      rather than strings for id labels.
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string or function that has scalar input and output
                      functional form of the cost function;
                      'exp' | 'pow' | custom function
    d_vars          : array (optional)
                      n x p; p attributes for each destination of n flows;
                      default is None
    constant        : boolean
                      True to include intercept in model; false by default
    framework       : string
                      estimation technique; currently only 'GLM' is avaialble
    Quasi           : boolean
                      True to estimate QuasiPoisson model; should result in same
                      parameters as Poisson but with altered covariance; default
                      to true which estimates Poisson model
    SF              : array
                      n x 1; eigenvector spatial filter to include in the model;
                      default to None which does not include a filter; not yet
                      implemented
    CD              : array
                      n x 1; competing destination term that accounts for the
                      likelihood that alternative destinations are considered
                      along with each destination under consideration for every
                      OD pair; defaults to None which does not include a CD
                      term; not yet implemented
    Lag             : W object
                      spatial weight for n observations (OD pairs) used to
                      construct a spatial autoregressive model and estimator;
                      defaults to None which does not include an autoregressive
                      term; not yet implemented

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    k               : integer
                      number of parameters
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    o               : array
                      n x 1; index of origin id's
    dv              : array 
                      n x p; p attributes for each destination of n flows
    constant        : boolean
                      True to include intercept in model; false by default
    y               : array
                      n x 1; dependent variable used in estimation including any
                      transformations
    X               : array
                      n x k, design matrix used in estimation
    params          : array
                      n x k, k estimated beta coefficients; k = # of origins + p + 1
    yhat            : array
                      n x 1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k x 1, standard errors of betas
    pvalues         : array
                      k x 1, two-tailed pvalues of parameters
    tvalues         : array
                      k x 1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    resid_dev       : array
                      n x 1, residual deviance of model
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    llnull          : float
                      value of the loglikelihood function evaluated with only an
                      intercept; see family.py for distribution-specific
                      loglikelihoods
    aic             : float 
                      Akaike information criterion
    D2              : float
                      percentage of explained deviance
    adj_D2          : float
                      adjusted percentage of explained deviance
    pseudo_R2       : float
                      McFadden's pseudo R2  (coefficient of determination) 
    adj_pseudoR2    : float
                      adjusted McFadden's pseudo R2
    SRMSE           : float
                      standardized root mean square error
    SSI             : float
                      Sorensen similarity index
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------

    >>> import numpy as np
    >>> import pysal
    >>> from pysal.contrib.spint.gravity import Production
    >>> db = pysal.open(pysal.examples.get_path('nyc_bikes_ct.csv'))
    >>> cost = np.array(db.by_col('tripduration')).reshape((-1,1))
    >>> flows = np.array(db.by_col('count')).reshape((-1,1))
    >>> o = np.array(db.by_col('o_tract')).reshape((-1,1))
    >>> d_cap = np.array(db.by_col('d_cap')).reshape((-1,1))
    >>> model = Production(flows, o, d_cap, cost, 'exp')
    >>> model.params[-4:]
    array([  5.38580065e+00,   5.00216058e+00,   8.55357745e-01,
            -2.27444394e-03])

    """
    def __init__(self, flows, origins, d_vars, cost, cost_func, constant=False,
            framework='GLM', SF=None, CD=None, Lag=None, Quasi=False):
        self.constant = constant
        self.f = self.reshape(flows)
        self.o = self.reshape(origins)
        
        try:
            if d_vars.shape[1]:
                p = d_vars.shape[1]
        except:
            p = 1
        self.dv = np.reshape(d_vars, (-1,p))
        self.c = self.reshape(cost)
        #User.check_arrays(self.f, self.o, self.dv, self.c)
       
        BaseGravity.__init__(self, self.f, self.c, cost_func=cost_func, d_vars=self.dv,
                origins=self.o, constant=constant, framework=framework,
                SF=SF, CD=CD, Lag=Lag, Quasi=Quasi)
    
    def local(self, locs=None):
        """
        Calibrate local models for subsets of data from a single location to all
        other locations
        
        Parameters
        ----------
        locs        : iterable of location (origins) labels; default is
                      None which calibrates a local model for each origin

        Returns
        -------
        results     : dict where keys are names of model outputs and diagnostics
                      and values are lists of location specific values
        """
        results = {}
        covs = self.dv.shape[1] + 1
        results['aic'] = []
        results['deviance'] = []
        results['pseudoR2'] = []
        results['adj_pseudoR2'] = []
        results['D2'] = []
        results['adj_D2'] = []
        results['SSI'] = []
        results['SRMSE'] = []
        for cov in range(covs):
            results['param' + str(cov)] = []
            results['pvalue' + str(cov)] = []
            results['tvalue' + str(cov)] = []
        if locs is None:
        	locs = np.unique(self.o)
        for loc in np.unique(locs):
            subset = self.o == loc
            f = self.reshape(self.f[subset])
            o = self.reshape(self.o[subset])
            d_vars = self.dv[subset.reshape(self.dv.shape[0]),:]
            dij = self.reshape(self.c[subset])
            model = Production(f, o, d_vars, dij, self.cf)
            results['aic'].append(model.aic)
            results['deviance'].append(model.deviance)
            results['pseudoR2'].append(model.pseudoR2)
            results['adj_pseudoR2'].append(model.adj_pseudoR2)
            results['D2'].append(model.D2)
            results['adj_D2'].append(model.adj_D2)
            results['SSI'].append(model.SSI)
            results['SRMSE'].append(model.SRMSE)
            for cov in range(covs):
                results['param' + str(cov)].append(model.params[cov])
                results['pvalue' + str(cov)].append(model.pvalues[cov])
                results['tvalue' + str(cov)].append(model.tvalues[cov])
        return results

class Attraction(BaseGravity):
    """
    Attraction-constrained (destination-constrained) gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    destinations    : array of strings
                      n x 1; unique identifiers of destinations of n flows; when
                      there are many destinations it will be faster to use
                      integers over strings for id labels.
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string or function that has scalar input and output
                      functional form of the cost function;
                      'exp' | 'pow' | custom function
    o_vars          : array (optional)
                      n x p; p attributes for each origin of  n flows; default
                      is None
    constant        : boolean
                      True to include intercept in model; false by default
    y               : array
                      n x 1; dependent variable used in estimation including any
                      transformations
    X               : array
                      n x k, design matrix used in estimation
    framework       : string
                      estimation technique; currently only 'GLM' is avaialble
    Quasi           : boolean
                      True to estimate QuasiPoisson model; should result in same
                      parameters as Poisson but with altered covariance; default
                      to true which estimates Poisson model
    SF              : array
                      n x 1; eigenvector spatial filter to include in the model;
                      default to None which does not include a filter; not yet
                      implemented
    CD              : array
                      n x 1; competing destination term that accounts for the
                      likelihood that alternative destinations are considered
                      along with each destination under consideration for every
                      OD pair; defaults to None which does not include a CD
                      term; not yet implemented
    Lag             : W object
                      spatial weight for n observations (OD pairs) used to
                      construct a spatial autoregressive model and estimator;
                      defaults to None which does not include an autoregressive
                      term; not yet implemented

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    k               : integer
                      number of parameters
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    d               : array
                      n x 1; index of destination id's
    ov              : array
                      n x p; p attributes for each origin of n flows
    constant        : boolean
                      True to include intercept in model; false by default
    params          : array
                      n x k, k estimated beta coefficients; k = # of
                      destinations + p + 1
    yhat            : array
                      n x 1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k x 1, standard errors of betas
    pvalues         : array
                      k x 1, two-tailed pvalues of parameters
    tvalues         : array
                      k x 1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    resid_dev       : array
                      n x 1, residual deviance of model
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    llnull          : float
                      value of the loglikelihood function evaluated with only an
                      intercept; see family.py for distribution-specific
                      loglikelihoods
    aic             : float 
                      Akaike information criterion
    D2              : float
                      percentage of explained deviance
    adj_D2          : float
                      adjusted percentage of explained deviance
    pseudo_R2       : float
                      McFadden's pseudo R2  (coefficient of determination) 
    adj_pseudoR2    : float
                      adjusted McFadden's pseudo R2
    SRMSE           : float
                      standardized root mean square error
    SSI             : float
                      Sorensen similarity index
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    >>> import numpy as np
    >>> import pysal
    >>> from pysal.contrib.spint.gravity import Attraction
    >>> db = pysal.open(pysal.examples.get_path('nyc_bikes_ct.csv'))
    >>> cost = np.array(db.by_col('tripduration')).reshape((-1,1))
    >>> flows = np.array(db.by_col('count')).reshape((-1,1))
    >>> d = np.array(db.by_col('d_tract')).reshape((-1,1))
    >>> o_cap = np.array(db.by_col('o_cap')).reshape((-1,1))
    >>> model = Attraction(flows, d, o_cap, cost, 'exp')
    >>> model.params[-4:]
    array([  5.23366116e+00,   4.89037868e+00,   8.82909095e-01,
            -2.29081323e-03])

    """
    def __init__(self, flows, destinations, o_vars, cost, cost_func,
            constant=False, framework='GLM', SF=None, CD=None, Lag=None,
            Quasi=False):
        self.f = np.reshape(flows, (-1,1))
        if len(o_vars.shape) > 1:
            p = o_vars.shape[1]
        else:
            p = 1
        self.ov = np.reshape(o_vars, (-1,p))
        self.d = np.reshape(destinations, (-1,1))
        self.c = np.reshape(cost, (-1,1))
        #User.check_arrays(self.f, self.d, self.ov, self.c)

        BaseGravity.__init__(self, self.f, self.c, cost_func=cost_func, o_vars=self.ov,
                 destinations=self.d, constant=constant,
                 framework=framework, SF=SF, CD=CD, Lag=Lag, Quasi=Quasi)

    def local(self, locs=None):
        """
        Calibrate local models for subsets of data from a single location to all
        other locations

        Parameters
        ----------
        locs        : iterable of location (destinations) labels; default is
                      None which calibrates a local model for each destination

        Returns
        -------
        results     : dict where keys are names of model outputs and diagnostics
                      and values are lists of location specific values
        """
        results = {}
        covs = self.ov.shape[1] + 1
        results['aic'] = []
        results['deviance'] = []
        results['pseudoR2'] = []
        results['adj_pseudoR2'] = []
        results['D2'] = []
        results['adj_D2'] = []
        results['SSI'] = []
        results['SRMSE'] = []
        for cov in range(covs):
            results['param' + str(cov)] = []
            results['pvalue' + str(cov)] = []
            results['tvalue' + str(cov)] = []
        if locs is  None:
        	locs = np.unique(self.d)
        for loc in np.unique(locs):
            subset = self.d == loc
            f = self.reshape(self.f[subset])
            d = self.reshape(self.d[subset])
            o_vars = self.ov[subset.reshape(self.ov.shape[0]),:]
            dij = self.reshape(self.c[subset])
            model = Attraction(f, d, o_vars, dij, self.cf)
            results['aic'].append(model.aic)
            results['deviance'].append(model.deviance)
            results['pseudoR2'].append(model.pseudoR2)
            results['adj_pseudoR2'].append(model.adj_pseudoR2)
            results['D2'].append(model.D2)
            results['adj_D2'].append(model.adj_D2)
            results['SSI'].append(model.SSI)
            results['SRMSE'].append(model.SRMSE)
            for cov in range(covs):
                results['param' + str(cov)].append(model.params[cov])
                results['pvalue' + str(cov)].append(model.pvalues[cov])
                results['tvalue' + str(cov)].append(model.tvalues[cov])
        return results

class Doubly(BaseGravity):
    """
    Doubly-constrained gravity-type spatial interaction model
    
    Parameters
    ----------
    flows           : array of integers
                      n x 1; observed flows between O origins and D destinations
    origins         : array of strings
                      n x 1; unique identifiers of origins of n flows; when
                      there are many origins it will be faster to use integers
                      rather than strings for id labels.
    destinations    : array of strings
                      n x 1; unique identifiers of destinations of n flows; when
                      there are many destinations it will be faster to use
                      integers rather than strings for id labels
    cost            : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cost_func       : string or function that has scalar input and output
                      functional form of the cost function;
                      'exp' | 'pow' | custom function
    constant        : boolean
                      True to include intercept in model; false by default
    y               : array
                      n x 1; dependent variable used in estimation including any
                      transformations
    X               : array
                      n x k, design matrix used in estimation
    framework       : string
                      estimation technique; currently only 'GLM' is avaialble
    Quasi           : boolean
                      True to estimate QuasiPoisson model; should result in same
                      parameters as Poisson but with altered covariance; default
                      to true which estimates Poisson model
    SF              : array
                      n x 1; eigenvector spatial filter to include in the model;
                      default to None which does not include a filter; not yet
                      implemented
    CD              : array
                      n x 1; competing destination term that accounts for the
                      likelihood that alternative destinations are considered
                      along with each destination under consideration for every
                      OD pair; defaults to None which does not include a CD
                      term; not yet implemented
    Lag             : W object
                      spatial weight for n observations (OD pairs) used to
                      construct a spatial autoregressive model and estimator;
                      defaults to None which does not include an autoregressive
                      term; not yet implemented

    Attributes
    ----------
    f               : array
                      n x 1; observed flows; dependent variable; y
    n               : integer
                      number of observations
    k               : integer
                      number of parameters
    c               : array 
                      n x 1; cost to overcome separation between each origin and
                      destination associated with a flow; typically distance or time
    cf              : function
                      cost function; used to transform cost variable
    o               : array
                      n x 1; index of origin id's
    d               : array
                      n x 1; index of destination id's
    constant        : boolean
                      True to include intercept in model; false by default
    params          : array
                      n x k, estimated beta coefficients; k = # of origins + #
                      of destinations; the first x-1 values
                      pertain to the x destinations (leaving out the first
                      destination to avoid perfect collinearity; no fixed
                      effect), the next x values pertain to the x origins, and the
                      final value is the distance decay coefficient
    yhat            : array
                      n x 1, predicted value of y (i.e., fittedvalues)
    cov_params      : array
                      Variance covariance matrix (kxk) of betas
    std_err         : array
                      k x 1, standard errors of betas
    pvalues         : array
                      k x 1, two-tailed pvalues of parameters
    tvalues         : array
                      k x 1, the tvalues of the standard errors
    deviance        : float
                      value of the deviance function evalued at params;
                      see family.py for distribution-specific deviance
    resid_dev       : array
                      n x 1, residual deviance of model
    llf             : float
                      value of the loglikelihood function evalued at params;
                      see family.py for distribution-specific loglikelihoods
    llnull          : float
                      value of the loglikelihood function evaluated with only an
                      intercept; see family.py for distribution-specific
                      loglikelihoods
    aic             : float 
                      Akaike information criterion
    D2              : float
                      percentage of explained deviance
    adj_D2          : float
                      adjusted percentage of explained deviance
    pseudo_R2       : float
                      McFadden's pseudo R2  (coefficient of determination) 
    adj_pseudoR2    : float
                      adjusted McFadden's pseudo R2
    SRMSE           : float
                      standardized root mean square error
    SSI             : float
                      Sorensen similarity index
    results         : object
                      Full results from estimated model. May contain addtional
                      diagnostics
    Example
    -------
    >>> import numpy as np
    >>> import pysal
    >>> from pysal.contrib.spint.gravity import Doubly
    >>> db = pysal.open(pysal.examples.get_path('nyc_bikes_ct.csv'))
    >>> cost = np.array(db.by_col('tripduration')).reshape((-1,1))
    >>> flows = np.array(db.by_col('count')).reshape((-1,1))
    >>> d = np.array(db.by_col('d_tract')).reshape((-1,1))
    >>> o = np.array(db.by_col('o_tract')).reshape((-1,1))
    >>> model = Doubly(flows, o, d, cost, 'exp')
    >>> model.params[-1:]
    array([-0.00232112])

    """
    def __init__(self, flows, origins, destinations, cost, cost_func,
            constant=False, framework='GLM', SF=None, CD=None, Lag=None,
            Quasi=False):

        self.f = np.reshape(flows, (-1,1))
        self.o = np.reshape(origins, (-1,1))
        self.d = np.reshape(destinations, (-1,1))
        self.c = np.reshape(cost, (-1,1))
        #User.check_arrays(self.f, self.o, self.d, self.c)

        BaseGravity.__init__(self, self.f, self.c, cost_func=cost_func, origins=self.o, 
                destinations=self.d, constant=constant,
                framework=framework, SF=SF, CD=CD, Lag=Lag, Quasi=Quasi)

    def local(self, locs=None):
        """
        **Not inmplemented for doubly-constrained models** Not possible due to
        insufficient degrees of freedom.

        Calibrate local models for subsets of data from a single location to all
        other locations
        """
        raise NotImplementedError("Local models not possible for"
        " doubly-constrained model due to insufficient degrees of freedom.")
