"""Ordinary Least Squares regression classes."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu"
import numpy as np
import numpy.linalg as la
import user_output as USER

__all__ = ["OLS"]

class RegressionProps:
    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    utu     : float
              Sum of the squared residuals
    sig2n    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable
              
    """

    @property
    def utu(self):
        if 'utu' not in self._cache:
            self._cache['utu'] = np.sum(self.u**2)
        return self._cache['utu']
    @property
    def sig2n(self):
        if 'sig2n' not in self._cache:
            self._cache['sig2n'] = self.utu / self.n
        return self._cache['sig2n']
    @property
    def sig2n_k(self):
        if 'sig2n_k' not in self._cache:
            self._cache['sig2n_k'] = self.utu / (self.n-self.k)
        return self._cache['sig2n_k']
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.xtxi)
        return self._cache['vm']
    
    @property
    def mean_y(self):
        if 'mean_y' not in self._cache:
            self._cache['mean_y']=np.mean(self.y)
        return self._cache['mean_y']
    @property
    def std_y(self):
        if 'std_y' not in self._cache:
            self._cache['std_y']=np.std(self.y, ddof=1)
        return self._cache['std_y']
    


class BaseOLS(RegressionProps):
    """
    Compute ordinary least squares (note: no consistency checks or
    diagnostics)

    Parameters
    ----------
    y        : array
               nx1 array of dependent variable
    x        : array
               nxk array of independent variables (assumed to be aligned with y)
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)

    Attributes
    ----------

    y       : array
              nx1 array of dependent variable
    x       : array
              nxk array of independent variables (with constant added if
              constant parameter set to True)
    betas   : array
              kx1 array with estimated coefficients
    xtx     : array
              kxk array
    xtxi    : array
              kxk array of inverted xtx
    u       : array
              nx1 array of residuals
    predy   : array
              nx1 array of predicted values
    n       : int
              Number of observations
    k       : int
              Number of variables (constant included)
    utu     : float
              Sum of the squared residuals
    sig2    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable
              
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols=BaseOLS(y,X)
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self, y, x, constant=True):
        if constant:
            x = np.hstack((np.ones(y.shape), x))
        self.set_x(x)
        xty = np.dot(x.T, y)
        self.betas = np.dot(self.xtxi, xty)
        predy = np.dot(x, self.betas)
        u = y-predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = x.shape
        RegressionProps()
        self._cache = {}
        self.sig2 = self.sig2n_k

    def set_x(self, x):
        self.x = x
        self.xtx = np.dot(self.x.T, self.x)
        self.xtxi = la.inv(self.xtx)

class OLS(BaseOLS, USER.DiagnosticBuilder):
    """
    Compute ordinary least squares and return results and diagnostics.
    
    Parameters
    ----------

    y        : array
               nx1 array of dependent variable
    x        : array
               nxk array of independent variables (assumed to be aligned with y)
    w        : spatial weights object
               if provided then spatial diagnostics are computed
    constant : boolean
               If true it appends a vector of ones to the independent variables
               to estimate intercept (set to True by default)
    name_y   : string
               Name of dependent variables for use in output
    name_x   : list of strings
               Names of independent variables for use in output
    name_ds  : string
               Name of dataset for use in output
    vm       : boolean
               If True, include variance matrix in summary results
    pred     : boolean
               If True, include y, predicted values and residuals in summary results
    

    Attributes
    ----------

    y        : array
               nx1 array of dependent variable
    x        : array
               nxk array of independent variables (with constant added if
               constant parameter set to True)
    betas    : array
               kx1 array with estimated coefficients
    u        : array
               nx1 array of residuals
    predy    : array
               nx1 array of predicted values
    n        : int
               Number of observations
    k        : int
               Number of variables (constant included)
    name_ds  : string
               dataset's name
    name_y   : string
               Dependent variable's name
    name_x   : tuple
               Independent variables' names
    mean_y   : float
               Mean value of dependent variable
    std_y    : float
               Standard deviation of dependent variable
    vm       : array
               Variance covariance matrix (kxk)
    r2       : float
               R squared
    ar2      : float
               Adjusted R squared
    utu      : float
               Sum of the squared residuals
    sig2     : float
               Sigma squared
    sig2ML   : float
               Sigma squared ML 
    f_stat   : tuple
               Statistic (float), p-value (float)
    logll    : float
               Log likelihood        
    aic      : float
               Akaike info criterion 
    schwarz  : float
               Schwarz info criterion     
    std_err  : array
               1xk array of Std.Error    
    t_stat   : list of tuples
               Each tuple contains the pair (statistic, p-value), where each is
               a float; same order as self.x
    mulColli : float
               Multicollinearity condition number
    jarque_bera : dictionary
               'jb': Jarque-Bera statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    breusch_pagan : dictionary
               'bp': Breusch-Pagan statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    koenker_bassett : dictionary
               'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    white    : dictionary
               'wh': White statistic (float); 'pvalue': p-value (float); 'df':
               degrees of freedom (int)  
    lm_error : tuple
               Lagrange multiplier test for spatial error model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    lm_lag   : tuple
               Lagrange multiplier test for spatial lag model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    rlm_error : tuple
               Robust lagrange multiplier test for spatial error model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    rlm_lag   : tuple
               Robust lagrange multiplier test for spatial lag model; each tuple 
               contains the pair (statistic, p-value), where each is a float; only 
               available if w defined
    lm_sarma : tuple
               Lagrange multiplier test for spatial SARMA model; each tuple contains
               the pair (statistic, p-value), where each is a float; only available 
               if w defined
    moran_res : tuple
                Tuple containing the triple (Moran's I, stansardized Moran's
                I, p-value); only available if w defined
    summary  : string
               Including all the information in OLS class in nice format          
     
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("../examples/columbus.dbf","r")
    >>> # Extract CRIME column from the dbf file
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(y, X, name_y='crime', name_x=['inc','hoval'], name_ds='columbus')
    >>> ols.betas
    array([[ 68.6189611 ],
           [ -1.59731083],
           [ -0.27393148]])
    
    """
    def __init__(self, y, x, w=None, constant=True, name_y=None, name_x=None,\
                        name_ds=None, vm=False, pred=False):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseOLS.__init__(self, y, x, constant) 
        self.title = "ORDINARY LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        USER.DiagnosticBuilder.__init__(self, x=x, constant=constant, w=w,\
                                            vm=vm, pred=pred)

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



