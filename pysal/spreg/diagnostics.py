"""
Diagnostics for regression estimations. 
        
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Nicholas Malizia nicholas.malizia@asu.edu "

import pysal
from pysal.common import *
from math import sqrt

__all__ = [ "f_stat", "t_stat", "r2", "ar2", "se_betas", "log_likelihood", "akaike", "schwarz", "condition_index", "jarque_bera", "breusch_pagan", "white", "koenker_bassett", "vif" ]

def f_stat(reg):
    """
    Calculates the f-statistic and associated p-value of the regression.
    (For two stage least squares see f_stat_tsls)
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    fs_result       : tuple
                      includes value of F statistic and associated p-value

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.f_stat(reg)
    >>> print("%12.12f"%testresult[0],"%12.12f"%testresult[1])
    ('28.385629224695', '0.000000009341')

    """ 
    k = reg.k            # (scalar) number of ind. vars (includes constant)
    n = reg.n            # (scalar) number of observations
    utu = reg.utu        # (scalar) residual sum of squares
    predy = reg.predy    # (array) vector of predicted values (n x 1)
    mean_y = reg.mean_y  # (scalar) mean of dependent observations
    Q = utu
    U = np.sum((predy-mean_y)**2)
    fStat = (U/(k-1))/(Q/(n-k))
    pValue = stats.f.sf(fStat,k-1,n-k)
    fs_result = (fStat, pValue)
    return fs_result






def t_stat(reg, z_stat=False):
    """
    Calculates the t-statistics (or z-statistics) and associated p-values.
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
    z_stat          : boolean
                      If True run z-stat instead of t-stat
        
    Returns
    -------    
    ts_result       : list of tuples
                      each tuple includes value of t statistic (or z
                      statistic) and associated p-value

    References
    ----------

    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> # t-stat for OLS
    >>> testresult = diagnostics.t_stat(reg)
    >>> print("%12.12f"%testresult[0][0], "%12.12f"%testresult[0][1], "%12.12f"%testresult[1][0], "%12.12f"%testresult[1][1], "%12.12f"%testresult[2][0], "%12.12f"%testresult[2][1])
    ('14.490373143689', '0.000000000000', '-4.780496191297', '0.000018289595', '-2.654408642718', '0.010874504910')
    """ 
    
    k = reg.k           # (scalar) number of ind. vars (includes constant)
    n = reg.n           # (scalar) number of observations
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)
    betas = reg.betas   # (array) coefficients of the regressors (1 x k) 
    variance = vm.diagonal()
    tStat = betas.reshape(len(betas),)/ np.sqrt(variance)
    ts_result = []
    for t in tStat:
        if z_stat:
            ts_result.append((t, stats.norm.sf(abs(t))*2))
        else:
            ts_result.append((t, stats.t.sf(abs(t),n-k)*2))
    return ts_result




def r2(reg):
    """
    Calculates the R^2 value for the regression. 
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    r2_result       : float
                      value of the coefficient of determination for the
                      regression 

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.r2(reg)
    >>> testresult
    0.55240404083742334
    
    """ 
    y = reg.y               # (array) vector of dep observations (n x 1)
    mean_y = reg.mean_y     # (scalar) mean of dep observations
    utu = reg.utu           # (scalar) residual sum of squares
    ss_tot = sum((y-mean_y)**2)
    r2 = 1-utu/ss_tot
    r2_result = r2[0]
    return r2_result



def ar2(reg):
    """
    Calculates the adjusted R^2 value for the regression. 
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model   

    Returns
    ----------
    ar2_result      : float
                      value of R^2 adjusted for the number of explanatory
                      variables.

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.ar2(reg)
    >>> testresult
    0.5329433469607896

    """ 
    k = reg.k       # (scalar) number of ind. variables (includes constant)
    n = reg.n       # (scalar) number of observations
    ar2_result =  1-(1-r2(reg))*(n-1)/(n-k)
    return ar2_result



def se_betas(reg):
    """
    Calculates the standard error of the regression coefficients.
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    se_result       : array
                      includes standard errors of each coefficient (1 x k)

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 
    
    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.se_betas(reg)
    >>> testresult
    array([ 4.73548613,  0.33413076,  0.10319868])
    
    """ 
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)  
    variance = vm.diagonal()
    se_result = np.sqrt(variance)
    return se_result



def log_likelihood(reg):
    """
    Calculates the log-likelihood value for the regression. 
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    ll_result       : float
                      value for the log-likelihood of the regression.

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.log_likelihood(reg)
    >>> testresult
    -187.3772388121491

    """
    n = reg.n       # (scalar) number of observations
    utu = reg.utu   # (scalar) residual sum of squares
    ll_result = -0.5*(n*(np.log(2*math.pi))+n*np.log(utu/n)+(utu/(utu/n)))
    return ll_result   



def akaike(reg):
    """
    Calculates the Akaike Information Criterion

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    aic_result      : scalar
                      value for Akaike Information Criterion of the
                      regression. 

    References
    ----------
    .. [1] H. Akaike. 1974. A new look at the statistical identification
       model. IEEE Transactions on Automatic Control, 19(6):716-723.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.akaike(reg)
    >>> testresult
    380.7544776242982

    """
    n = reg.n       # (scalar) number of observations
    k = reg.k       # (scalar) number of ind. variables (including constant)
    utu = reg.utu   # (scalar) residual sum of squares
    aic_result = 2*k + n*(np.log((2*np.pi*utu)/n)+1)
    return aic_result



def schwarz(reg):
    """
    Calculates the Schwarz Information Criterion

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    bic_result      : scalar
                      value for Schwarz (Bayesian) Information Criterion of
                      the regression. 

    References
    ----------
    .. [1] G. Schwarz. 1978. Estimating the dimension of a model. The
       Annals of Statistics, pages 461-464. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.schwarz(reg)
    >>> testresult
    386.42993851863008

    """
    n = reg.n       # (scalar) number of observations
    k = reg.k       # (scalar) number of ind. variables (including constant)
    utu = reg.utu   # (scalar) residual sum of squares
    sc_result = k*np.log(n) + n*(np.log((2*np.pi*utu)/n)+1)
    return sc_result



def condition_index(reg):
    """
    Calculates the multicollinearity condition index according to Belsey,
    Kuh and Welsh (1980)

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    ci_result       : float
                      scalar value for the multicollinearity condition
                      index. 

    References
    ----------
    .. [1] D. Belsley, E. Kuh, and R. Welsch. 1980. Regression Diagnostics. 
       New York: Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.condition_index(reg)
    >>> print("%12.12f"%testresult)
    6.541827751444

    """
    xtx = reg.xtx   # (array) k x k projection matrix (includes constant)
    diag = np.diagonal(xtx)
    scale = xtx/diag    
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_result = sqrt(max_eigval/min_eigval)
    return ci_result



def jarque_bera(reg):
    """
    Jarque-Bera test for normality in the residuals. 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ------- 
    jb_result       : dictionary
                      contains the statistic (jb) for the Jarque-Bera test
                      and the associated p-value (p-value)
    df              : integer
                      degrees of freedom associated with the test (always 2)
    jb              : float
                      value of the test statistic
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with 2 df)

    References
    ----------
    .. [1] C. Jarque and A. Bera. 1980. Efficient tests for normality,
       homoscedasticity and serial independence of regression residuals.
       Economics Letters, 6(3):255-259.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.jarque_bera(reg)
    >>> testresult['df']
    2
    >>> print("%12.12f"%testresult['jb'])
    1.835752520076
    >>> print("%12.12f"%testresult['pvalue'])
    0.399366291249

    """
    n = reg.n               # (scalar) number of observations
    u = reg.u               # (array) residuals from regression 
    u2 = u**2                          
    u3 = u**3                          
    u4 = u**4                           
    mu2 = np.mean(u2)       
    mu3 = np.mean(u3)       
    mu4 = np.mean(u4)         
    S = mu3/(mu2**(1.5))    # skewness measure
    K = (mu4/(mu2**2))      # kurtosis measure
    jb = n*(((S**2)/6)+((K-3)**2)/24)
    pvalue=stats.chisqprob(jb,2)
    jb_result={"df":2,"jb":jb,'pvalue':pvalue}
    return jb_result 



def breusch_pagan(reg):
    """
    Calculates the Breusch-Pagan test statistic to check for
    heteroskedasticity. 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model 

    Returns
    -------
    bp_result       : dictionary
                      contains the statistic (bp) for the Breusch-Pagan test
                      and the associated p-value (p-value)
    bp              : float
                      scalar value for the Breusch-Pagan test statistic.
    df              : integer
                      degrees of freedom associated with the test (k)
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with k df)

    References
    ----------
    
    .. [1] T. Breusch and A. Pagan. 1979. A simple test for
       heteroscedasticity and random coefficient variation. Econometrica:
       Journal of the Econometric Society, 47(5):1287-1294.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.breusch_pagan(reg)
    >>> testresult['df']
    2
    >>> print("%12.12f"%testresult['bp'])
    10.012849713094
    >>> print("%12.12f"%testresult['pvalue'])
    0.006694795426

    """
    e2 = reg.u**2
    e = reg.u
    n = reg.n
    x = reg.x
    k = reg.k
    ete = reg.utu
    constant = constant_check(x)

    den = ete/n
    g = e2/den - 1.0

    if constant == False: 
        z = np.hstack((np.ones((n,1)),x))
        df = k
    else:
        z = x
        df = k-1

    zt = np.transpose(z)
    gt = np.transpose(g)
    gtz = np.dot(gt,z)
    ztg = np.dot(zt,g)
    ztz = np.dot(zt,z)
    ztzi = la.inv(ztz)

    part1 = np.dot(gtz, ztzi)
    part2 = np.dot(part1,ztg)
    bp_array = 0.5*part2
    bp = bp_array[0,0]

    pvalue=stats.chisqprob(bp,df)
    bp_result={'df':df,'bp':bp, 'pvalue':pvalue}
    return bp_result



def white(reg):
    """
    Calculates the White test to check for heteroskedasticity. 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
    constant        : boolean
                      if true the original regression includes a constant,
                      set to "True" by default

    Returns
    -------
    white_result    : dictionary
                      contains the statistic (white), degrees of freedom
                      (df) and the associated p-value (pvalue) for the
                      White test. 
    white           : float
                      scalar value for the White test statistic.
    df              : integer
                      degrees of freedom associated with the test
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with k df)
    
    References
    ----------
    .. [1] H. White. 1980. A heteroskedasticity-consistent covariance matrix
       estimator and a direct test for heteroskdasticity. Econometrica.
       48(4) 817-838. 


    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.white(reg)
    >>> testresult['df']
    5
    >>> print("%12.12f"%testresult['wh'])
    19.946008239903
    >>> print("%12.12f"%testresult['pvalue'])
    0.001279222817

    """
    e = reg.u**2
    k = reg.k
    n = reg.n
    y = reg.y
    X = reg.x
    constant = constant_check(X)
    
    # Check for constant, if none add one, see Greene 2003, pg. 222
    if constant == False: 
        X = np.hstack((np.ones((n,1)),X))

    # Check for multicollinearity in the X matrix
    ci = condition_index(reg)
    if ci > 30:
        white_result = "N/A"
        return white_result

    # Compute cross-products of the regression variables
    A = []
    for i in range(k-1):
        for j in range(i+1,k):
            v = X[:,i]*X[:,j]
            A.append(v)
    
    # Square the regression variables
    for i in range(k):
        v = X[:,i]**2
        A.append(v)

    # Convert to an array with the proper dimensions and append the original
    # non-binary variables
    A = np.array(A).T
    A = np.hstack((X,A))
    n,k = A.shape

    # Check to identify any duplicate columns in A
    omitcolumn = []
    for i in range(k):
        current = A[:,i]
        for j in range(k):
            check = A[:,j]
            if i < j:
                test = abs(current - check).sum()
                if test == 0:
                    omitcolumn.append(j)

    uniqueomit = set(omitcolumn)
    omitcolumn = list(uniqueomit)

    # Now the identified columns must be removed (done in reverse to prevent
    # renumbering)
    omitcolumn.reverse()
    for c in omitcolumn:
        A = np.delete(A,c,1)
    n,k = A.shape

    # Conduct the auxiliary regression and calculate the statistic
    import ols as OLS
    aux_reg = OLS.BaseOLS(e,A,constant=False)
    aux_r2 = r2(aux_reg)
    wh = aux_r2*n
    df = k-1
    pvalue = stats.chisqprob(wh,df)
    white_result={'df':df,'wh':wh, 'pvalue':pvalue}
    return white_result 



def koenker_bassett(reg):
    """
    Calculates the Koenker-Bassett test statistic to check for
    heteroskedasticity. 

    Parameters
    ----------
    reg             : regression output
                      output from an instance of a regression class

    Returns
    -------
    kb_result       : dictionary
                      contains the statistic (kb), degrees of freedom (df)
                      and the associated p-value (pvalue) for the test. 
    kb              : float
                      scalar value for the Koenker-Bassett test statistic.
    df              : integer
                      degrees of freedom associated with the test
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed)

    Reference
    ---------
    .. [1] R. Koenker and G. Bassett. 1982. Robust tests for
       heteroscedasticity based on regression quantiles. Econometrica,
       50(1):43-61. 

    .. [2] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.koenker_bassett(reg)
    >>> testresult['df']
    2
    >>> print("%12.12f"%testresult['kb'])
    7.216564472188
    >>> print("%12.12f"%testresult['pvalue'])
    0.027098355486

    """
    # The notation here matches that of Greene (2003).
    u = reg.u**2
    e = reg.u
    n = reg.n
    k = reg.k
    x = reg.x
    ete = reg.utu
    constant = constant_check(x)

    ubar = ete/n
    ubari = ubar*np.ones((n,1))
    g = u-ubari
    v = (1.0/n)*np.sum((u-ubar)**2)

    # This is required because the first column of z must be a constant.
    if constant == False: 
        z = np.hstack((np.ones((n,1)),x))
        df = k
    else:
        z = x
        df = k-1

    # Conduct the auxiliary regression.
    zt = np.transpose(z)
    gt = np.transpose(g)
    gtz = np.dot(gt,z)
    ztg = np.dot(zt,g)
    ztz = np.dot(zt,z)
    ztzi = la.inv(ztz)

    part1 = np.dot(gtz, ztzi)
    part2 = np.dot(part1,ztg)
    kb_array = (1.0/v)*part2
    kb = kb_array[0,0]
    
    pvalue=stats.chisqprob(kb,df)
    kb_result = {'kb':kb,'df':df,'pvalue':pvalue}
    return kb_result



def vif(reg):
    """
    Calculates the variance inflation factor for each independent variable.
    For the ease of indexing the results, the constant is currently
    included. This should be omitted when reporting the results to the
    output text.
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
        
    Returns
    -------    
    vif_result      : list of tuples
                      each tuple includes the vif and the tolerance, the
                      order of the variables corresponds to their order in
                      the reg.x matrix

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> testresult = diagnostics.vif(reg)
    >>> incvif = testresult[1]
    >>> print("%12.12f"%incvif[0])
    1.333117497189
    >>> print("%12.12f"%incvif[1])
    0.750121427487
    >>> hovalvif = testresult[1]
    >>> print("%12.12f"%hovalvif[0])
    1.333117497189
    >>> print("%12.12f"%hovalvif[1])
    0.750121427487

    """
    X = reg.x
    n,k = X.shape
    vif_result = []

    for j in range(k):
        Z = X.copy()
        Z = np.delete(Z,j,1)
        y  = X[:,j]
        import ols as OLS
        aux = OLS.BaseOLS(y,Z,constant=False)
        mean_y = aux.mean_y
        utu = aux.utu
        ss_tot = sum((y-mean_y)**2)
        r2aux = 1-utu/ss_tot
        tolj = 1 - r2aux
        vifj = 1 / tolj
        resj = (vifj,tolj)
        vif_result.append(resj)
    return vif_result



def constant_check(array):
    """
    Checks to see numpy array includes a constant.

    Parameters
    ----------
    array           : array
                      an array of variables to be inspected 

    Returns
    -------
    constant        : boolean
                      true signifies the presence of a constant

    Example
    -------

    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
    >>> db = pysal.open("../examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> diagnostics.constant_check(reg.x)
    True


    """
    n,k = array.shape
    constant = False
    for j in range(k):
        variable = array[:,j]
        variable = variable.ravel()
        test = set(variable)
        test = list(test)
        if len(test) == 1:
            constant = True
            break
    return constant
        

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

