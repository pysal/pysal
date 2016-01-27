"""
Diagnostics for regression estimations. 
        
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Nicholas Malizia nicholas.malizia@asu.edu "

import pysal
from pysal.common import *
import scipy.sparse as SP
from math import sqrt
from utils import spmultiply, sphstack, spmin, spmax


__all__ = [
    "f_stat", "t_stat", "r2", "ar2", "se_betas", "log_likelihood", "akaike", "schwarz",
    "condition_index", "jarque_bera", "breusch_pagan", "white", "koenker_bassett", "vif", "likratiotest"]


def f_stat(reg):
    """
    Calculates the f-statistic and associated p-value of the
    regression. [Greene2003]_
    (For two stage least squares see f_stat_tsls)

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    fs_result       : tuple
                      includes value of F statistic and associated p-value

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data. 

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.    

    >>> reg = OLS(y,X)

    Calculate the F-statistic for the regression. 

    >>> testresult = diagnostics.f_stat(reg)

    Print the results tuple, including the statistic and its significance.

    >>> print("%12.12f"%testresult[0],"%12.12f"%testresult[1])
    ('28.385629224695', '0.000000009341')

    """
    k = reg.k            # (scalar) number of ind. vars (includes constant)
    n = reg.n            # (scalar) number of observations
    utu = reg.utu        # (scalar) residual sum of squares
    predy = reg.predy    # (array) vector of predicted values (n x 1)
    mean_y = reg.mean_y  # (scalar) mean of dependent observations
    Q = utu
    U = np.sum((predy - mean_y) ** 2)
    fStat = (U / (k - 1)) / (Q / (n - k))
    pValue = stats.f.sf(fStat, k - 1, n - k)
    fs_result = (fStat, pValue)
    return fs_result


def t_stat(reg, z_stat=False):
    """
    Calculates the t-statistics (or z-statistics) and associated
    p-values. [Greene2003]_

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

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data. 

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.    

    >>> reg = OLS(y,X)

    Calculate t-statistics for the regression coefficients. 

    >>> testresult = diagnostics.t_stat(reg)

    Print the tuples that contain the t-statistics and their significances.

    >>> print("%12.12f"%testresult[0][0], "%12.12f"%testresult[0][1], "%12.12f"%testresult[1][0], "%12.12f"%testresult[1][1], "%12.12f"%testresult[2][0], "%12.12f"%testresult[2][1])
    ('14.490373143689', '0.000000000000', '-4.780496191297', '0.000018289595', '-2.654408642718', '0.010874504910')
    """

    k = reg.k           # (scalar) number of ind. vars (includes constant)
    n = reg.n           # (scalar) number of observations
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)
    betas = reg.betas   # (array) coefficients of the regressors (1 x k)
    variance = vm.diagonal()
    tStat = betas[range(0, len(vm))].reshape(len(vm),) / np.sqrt(variance)
    ts_result = []
    for t in tStat:
        if z_stat:
            ts_result.append((t, stats.norm.sf(abs(t)) * 2))
        else:
            ts_result.append((t, stats.t.sf(abs(t), n - k) * 2))
    return ts_result


def r2(reg):
    """
    Calculates the R^2 value for the regression. [Greene2003]_ 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    r2_result       : float
                      value of the coefficient of determination for the
                      regression 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression. 

    >>> reg = OLS(y,X)

    Calculate the R^2 value for the regression. 

    >>> testresult = diagnostics.r2(reg)

    Print the result. 

    >>> print("%1.8f"%testresult)
    0.55240404

    """
    y = reg.y               # (array) vector of dep observations (n x 1)
    mean_y = reg.mean_y     # (scalar) mean of dep observations
    utu = reg.utu           # (scalar) residual sum of squares
    ss_tot = ((y - mean_y) ** 2).sum(0)
    r2 = 1 - utu / ss_tot
    r2_result = r2[0]
    return r2_result


def ar2(reg):
    """
    Calculates the adjusted R^2 value for the regression. [Greene2003]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model   

    Returns
    ----------
    ar2_result      : float
                      value of R^2 adjusted for the number of explanatory
                      variables.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression. 

    >>> reg = OLS(y,X)

    Calculate the adjusted R^2 value for the regression. 
    >>> testresult = diagnostics.ar2(reg)

    Print the result. 

    >>> print("%1.8f"%testresult)
    0.53294335

    """
    k = reg.k       # (scalar) number of ind. variables (includes constant)
    n = reg.n       # (scalar) number of observations
    ar2_result = 1 - (1 - r2(reg)) * (n - 1) / (n - k)
    return ar2_result


def se_betas(reg):
    """
    Calculates the standard error of the regression coefficients. [Greene2003]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    ----------
    se_result       : array
                      includes standard errors of each coefficient (1 x k)

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data. 

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression. 

    >>> reg = OLS(y,X)

    Calculate the standard errors of the regression coefficients. 

    >>> testresult = diagnostics.se_betas(reg)

    Print the vector of standard errors. 

    >>> testresult
    array([ 4.73548613,  0.33413076,  0.10319868])

    """
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)
    variance = vm.diagonal()
    se_result = np.sqrt(variance)
    return se_result


def log_likelihood(reg):
    """
    Calculates the log-likelihood value for the regression. [Greene2003]_ 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    ll_result       : float
                      value for the log-likelihood of the regression.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data. 

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the log-likelihood for the regression. 

    >>> testresult = diagnostics.log_likelihood(reg)

    Print the result. 

    >>> testresult
    -187.3772388121491

    """
    n = reg.n       # (scalar) number of observations
    utu = reg.utu   # (scalar) residual sum of squares
    ll_result = -0.5 * \
        (n * (np.log(2 * math.pi)) + n * np.log(utu / n) + (utu / (utu / n)))
    return ll_result


def akaike(reg):
    """
    Calculates the Akaike Information Criterion. [Akaike1974]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    aic_result      : scalar
                      value for Akaike Information Criterion of the
                      regression. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the Akaike Information Criterion (AIC). 

    >>> testresult = diagnostics.akaike(reg)

    Print the result. 

    >>> testresult
    380.7544776242982

    """
    k = reg.k       # (scalar) number of explanatory vars (including constant)
    try:   # ML estimation, logll already exists
        # spatial coefficient included in k
        aic_result = 2.0 * k - 2.0 * reg.logll
    except AttributeError:           # OLS case
        n = reg.n       # (scalar) number of observations
        utu = reg.utu   # (scalar) residual sum of squares
        aic_result = 2 * k + n * (np.log((2 * np.pi * utu) / n) + 1)
    return aic_result


def schwarz(reg):
    """
    Calculates the Schwarz Information Criterion. [Schwarz1978]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    bic_result      : scalar
                      value for Schwarz (Bayesian) Information Criterion of
                      the regression. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the Schwarz Information Criterion. 

    >>> testresult = diagnostics.schwarz(reg)

    Print the results. 

    >>> testresult
    386.42993851863008

    """
    n = reg.n      # (scalar) number of observations
    k = reg.k      # (scalar) number of ind. variables (including constant)
    try:  # ML case logll already computed
        # spatial coeff included in k
        sc_result = k * np.log(n) - 2.0 * reg.logll
    except AttributeError:          # OLS case
        utu = reg.utu  # (scalar) residual sum of squares
        sc_result = k * np.log(n) + n * (np.log((2 * np.pi * utu) / n) + 1)
    return sc_result


def condition_index(reg):
    """
    Calculates the multicollinearity condition index according to Belsey,
    Kuh and Welsh (1980) [Belsley1980]_.

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

    Returns
    -------
    ci_result       : float
                      scalar value for the multicollinearity condition
                      index. 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the condition index to check for multicollinearity.

    >>> testresult = diagnostics.condition_index(reg)

    Print the result.

    >>> print("%1.3f"%testresult)
    6.542

    """
    if hasattr(reg, 'xtx'):
        xtx = reg.xtx   # (array) k x k projection matrix (includes constant)
    elif hasattr(reg, 'hth'):
        xtx = reg.hth   # (array) k x k projection matrix (includes constant)
    diag = np.diagonal(xtx)
    scale = xtx / diag
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_result = sqrt(max_eigval / min_eigval)
    return ci_result


def jarque_bera(reg):
    """
    Jarque-Bera test for normality in the residuals. [Jarque1980]_ 

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
                      degrees of freedom for the test (always 2)
    jb              : float
                      value of the test statistic
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with 2 df)

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"), "r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the Jarque-Bera test for normality of residuals.

    >>> testresult = diagnostics.jarque_bera(reg)

    Print the degrees of freedom for the test.

    >>> testresult['df']
    2

    Print the test statistic. 

    >>> print("%1.3f"%testresult['jb'])
    1.836

    Print the associated p-value.

    >>> print("%1.4f"%testresult['pvalue'])
    0.3994

    """
    n = reg.n               # (scalar) number of observations
    u = reg.u               # (array) residuals from regression
    u2 = u ** 2
    u3 = u ** 3
    u4 = u ** 4
    mu2 = np.mean(u2)
    mu3 = np.mean(u3)
    mu4 = np.mean(u4)
    S = mu3 / (mu2 ** (1.5))    # skewness measure
    K = (mu4 / (mu2 ** 2))      # kurtosis measure
    jb = n * (((S ** 2) / 6) + ((K - 3) ** 2) / 24)
    pvalue = stats.chisqprob(jb, 2)
    jb_result = {"df": 2, "jb": jb, 'pvalue': pvalue}
    return jb_result


def breusch_pagan(reg, z=None):
    """
    Calculates the Breusch-Pagan test statistic to check for
    heteroscedasticity. [Breusch1979]_ 

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
    z               : array
                      optional input for specifying an alternative set of
                      variables (Z) to explain the observed variance. By
                      default this is a matrix of the squared explanatory
                      variables (X**2) with a constant added to the first
                      column if not already present. In the default case,
                      the explanatory variables are squared to eliminate
                      negative values. 

    Returns
    -------
    bp_result       : dictionary
                      contains the statistic (bp) for the test and the
                      associated p-value (p-value)
    bp              : float
                      scalar value for the Breusch-Pagan test statistic
    df              : integer
                      degrees of freedom associated with the test (k)
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with k df)

    Notes
    -----
    x attribute in the reg object must have a constant term included. This is
    standard for spreg.OLS so no testing done to confirm constant.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"), "r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the Breusch-Pagan test for heteroscedasticity.

    >>> testresult = diagnostics.breusch_pagan(reg)

    Print the degrees of freedom for the test.

    >>> testresult['df']
    2

    Print the test statistic.

    >>> print("%1.3f"%testresult['bp'])
    7.900

    Print the associated p-value. 

    >>> print("%1.4f"%testresult['pvalue'])
    0.0193

    """
    e2 = reg.u ** 2
    e = reg.u
    n = reg.n
    k = reg.k
    ete = reg.utu

    den = ete / n
    g = e2 / den - 1.0

    if z == None:
        x = reg.x
        #constant = constant_check(x)
        # if constant == False:
        #    z = np.hstack((np.ones((n,1)),x))**2
        # else:
        #    z = x**2
        z = spmultiply(x, x)
    else:
        #constant = constant_check(z)
        # if constant == False:
        #    z = np.hstack((np.ones((n,1)),z))
        pass

    n, p = z.shape

    # Check to identify any duplicate columns in Z
    omitcolumn = []
    for i in range(p):
        current = z[:, i]
        for j in range(p):
            check = z[:, j]
            if i < j:
                test = abs(current - check).sum()
                if test == 0:
                    omitcolumn.append(j)

    uniqueomit = set(omitcolumn)
    omitcolumn = list(uniqueomit)

    # Now the identified columns must be removed (done in reverse to
    # prevent renumbering)
    omitcolumn.sort()
    omitcolumn.reverse()
    for c in omitcolumn:
        z = np.delete(z, c, 1)
    n, p = z.shape

    df = p - 1

    # Now that the variables are prepared, we calculate the statistic
    zt = np.transpose(z)
    gt = np.transpose(g)
    gtz = np.dot(gt, z)
    ztg = np.dot(zt, g)
    ztz = np.dot(zt, z)
    ztzi = la.inv(ztz)

    part1 = np.dot(gtz, ztzi)
    part2 = np.dot(part1, ztg)
    bp_array = 0.5 * part2
    bp = bp_array[0, 0]

    pvalue = stats.chisqprob(bp, df)
    bp_result = {'df': df, 'bp': bp, 'pvalue': pvalue}
    return bp_result


def white(reg):
    """
    Calculates the White test to check for heteroscedasticity. [White1980]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model

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

    Notes
    -----
    x attribute in the reg object must have a constant term included. This is
    standard for spreg.OLS so no testing done to confirm constant.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the White test for heteroscedasticity.

    >>> testresult = diagnostics.white(reg)

    Print the degrees of freedom for the test.

    >>> print testresult['df']
    5

    Print the test statistic.

    >>> print("%1.3f"%testresult['wh'])
    19.946

    Print the associated p-value. 

    >>> print("%1.4f"%testresult['pvalue'])
    0.0013

    """
    e = reg.u ** 2
    k = int(reg.k)
    n = int(reg.n)
    y = reg.y
    X = reg.x
    #constant = constant_check(X)

    # Check for constant, if none add one, see Greene 2003, pg. 222
    # if constant == False:
    #    X = np.hstack((np.ones((n,1)),X))

    # Check for multicollinearity in the X matrix
    ci = condition_index(reg)
    if ci > 30:
        white_result = "Not computed due to multicollinearity."
        return white_result

    # Compute cross-products and squares of the regression variables
    if type(X).__name__ == 'ndarray':
        A = np.zeros((n, (k * (k + 1)) // 2))
    elif type(X).__name__ == 'csc_matrix' or type(X).__name__ == 'csr_matrix':
        # this is probably inefficient
        A = SP.lil_matrix((n, (k * (k + 1)) // 2))
    else:
        raise Exception, "unknown X type, %s" % type(X).__name__
    counter = 0
    for i in range(k):
        for j in range(i, k):
            v = spmultiply(X[:, i], X[:, j], False)
            A[:, counter] = v
            counter += 1

    # Append the original variables
    A = sphstack(X, A)   # note: this also converts a LIL to CSR
    n, k = A.shape

    # Check to identify any duplicate or constant columns in A
    omitcolumn = []
    for i in range(k):
        current = A[:, i]
        # remove all constant terms (will add a constant back later)
        if spmax(current) == spmin(current):
            omitcolumn.append(i)
            pass
        # do not allow duplicates
        for j in range(k):
            check = A[:, j]
            if i < j:
                test = abs(current - check).sum()
                if test == 0:
                    omitcolumn.append(j)
    uniqueomit = set(omitcolumn)
    omitcolumn = list(uniqueomit)

    # Now the identified columns must be removed
    if type(A).__name__ == 'ndarray':
        A = np.delete(A, omitcolumn, 1)
    elif type(A).__name__ == 'csc_matrix' or type(A).__name__ == 'csr_matrix':
        # this is probably inefficient
        keepcolumn = range(k)
        for i in omitcolumn:
            keepcolumn.remove(i)
        A = A[:, keepcolumn]
    else:
        raise Exception, "unknown A type, %s" % type(X).__name__
    A = sphstack(np.ones((A.shape[0], 1)), A)   # add a constant back in
    n, k = A.shape

    # Conduct the auxiliary regression and calculate the statistic
    import ols as OLS
    aux_reg = OLS.BaseOLS(e, A)
    aux_r2 = r2(aux_reg)
    wh = aux_r2 * n
    df = k - 1
    pvalue = stats.chisqprob(wh, df)
    white_result = {'df': df, 'wh': wh, 'pvalue': pvalue}
    return white_result


def koenker_bassett(reg, z=None):
    """
    Calculates the Koenker-Bassett test statistic to check for
    heteroscedasticity. [Koenker1982]_ [Greene2003]_

    Parameters
    ----------
    reg             : regression output
                      output from an instance of a regression class
    z               : array
                      optional input for specifying an alternative set of
                      variables (Z) to explain the observed variance. By
                      default this is a matrix of the squared explanatory
                      variables (X**2) with a constant added to the first
                      column if not already present. In the default case,
                      the explanatory variables are squared to eliminate
                      negative values. 

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

    Notes
    -----
    x attribute in the reg object must have a constant term included. This is
    standard for spreg.OLS so no testing done to confirm constant.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the Koenker-Bassett test for heteroscedasticity.

    >>> testresult = diagnostics.koenker_bassett(reg)

    Print the degrees of freedom for the test.

    >>> testresult['df']
    2

    Print the test statistic.

    >>> print("%1.3f"%testresult['kb'])
    5.694

    Print the associated p-value. 

    >>> print("%1.4f"%testresult['pvalue'])
    0.0580

    """
    # The notation here matches that of Greene (2003).
    u = reg.u ** 2
    e = reg.u
    n = reg.n
    k = reg.k
    x = reg.x
    ete = reg.utu
    #constant = constant_check(x)

    ubar = ete / n
    ubari = ubar * np.ones((n, 1))
    g = u - ubari
    v = (1.0 / n) * np.sum((u - ubar) ** 2)

    if z == None:
        x = reg.x
        #constant = constant_check(x)
        # if constant == False:
        #    z = np.hstack((np.ones((n,1)),x))**2
        # else:
        #    z = x**2
        z = spmultiply(x, x)
    else:
        #constant = constant_check(z)
        # if constant == False:
        #    z = np.hstack((np.ones((n,1)),z))
        pass

    n, p = z.shape

    # Check to identify any duplicate columns in Z
    omitcolumn = []
    for i in range(p):
        current = z[:, i]
        for j in range(p):
            check = z[:, j]
            if i < j:
                test = abs(current - check).sum()
                if test == 0:
                    omitcolumn.append(j)

    uniqueomit = set(omitcolumn)
    omitcolumn = list(uniqueomit)

    # Now the identified columns must be removed (done in reverse to
    # prevent renumbering)
    omitcolumn.sort()
    omitcolumn.reverse()
    for c in omitcolumn:
        z = np.delete(z, c, 1)
    n, p = z.shape

    df = p - 1

    # Conduct the auxiliary regression.
    zt = np.transpose(z)
    gt = np.transpose(g)
    gtz = np.dot(gt, z)
    ztg = np.dot(zt, g)
    ztz = np.dot(zt, z)
    ztzi = la.inv(ztz)

    part1 = np.dot(gtz, ztzi)
    part2 = np.dot(part1, ztg)
    kb_array = (1.0 / v) * part2
    kb = kb_array[0, 0]

    pvalue = stats.chisqprob(kb, df)
    kb_result = {'kb': kb, 'df': df, 'pvalue': pvalue}
    return kb_result


def vif(reg):
    """
    Calculates the variance inflation factor for each independent variable.
    For the ease of indexing the results, the constant is currently
    included. This should be omitted when reporting the results to the
    output text. [Greene2003]_

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

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS

    Read the DBF associated with the Columbus data.

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")

    Create the dependent variable vector. 

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Create the matrix of independent variables. 

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression.

    >>> reg = OLS(y,X)

    Calculate the variance inflation factor (VIF). 
    >>> testresult = diagnostics.vif(reg)

    Select the tuple for the income variable. 

    >>> incvif = testresult[1]

    Print the VIF for income. 

    >>> print("%12.12f"%incvif[0])
    1.333117497189

    Print the tolerance for income. 

    >>> print("%12.12f"%incvif[1])
    0.750121427487

    Repeat for the home value variable. 

    >>> hovalvif = testresult[2]
    >>> print("%12.12f"%hovalvif[0])
    1.333117497189
    >>> print("%12.12f"%hovalvif[1])
    0.750121427487

    """
    X = reg.x
    n, k = X.shape
    vif_result = []

    for j in range(k):
        Z = X.copy()
        Z = np.delete(Z, j, 1)
        y = X[:, j]
        import ols as OLS
        aux = OLS.BaseOLS(y, Z)
        mean_y = aux.mean_y
        utu = aux.utu
        ss_tot = sum((y - mean_y) ** 2)
        if ss_tot == 0:
            resj = pysal.MISSINGVALUE
        else:
            r2aux = 1 - utu / ss_tot
            tolj = 1 - r2aux
            vifj = 1 / tolj
            resj = (vifj, tolj)
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
    >>> from ols import OLS
    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
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

    n, k = array.shape
    constant = False
    for j in range(k):
        variable = array[:, j]
        varmin = variable.min()
        varmax = variable.max()
        if varmin == varmax:
            constant = True
            break
    return constant


def likratiotest(reg0, reg1):
    """
    Likelihood ratio test statistic [Greene2003]_

    Parameters
    ----------

    reg0         : regression object for constrained model (H0)
    reg1         : regression object for unconstrained model (H1)

    Returns
    -------

    likratio     : dictionary
                   contains the statistic (likr), the degrees of
                   freedom (df) and the p-value (pvalue)
    likr         : float
                   likelihood ratio statistic
    df           : integer
                   degrees of freedom
    p-value      : float
                   p-value

    Examples
    --------

    >>> import numpy as np
    >>> import pysal as ps
    >>> import scipy.stats as stats
    >>> import pysal.spreg.ml_lag as lag

    Use the baltim sample data set

    >>> db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w.transform = 'r'

    OLS regression

    >>> ols1 = ps.spreg.OLS(y,x)

    ML Lag regression

    >>> mllag1 = lag.ML_Lag(y,x,w)

    >>> lr = likratiotest(ols1,mllag1)

    >>> print "Likelihood Ratio Test: {0:.4f}       df: {1}        p-value: {2:.4f}".format(lr["likr"],lr["df"],lr["p-value"])
    Likelihood Ratio Test: 44.5721       df: 1        p-value: 0.0000

    """

    likratio = {}

    try:
        likr = 2.0 * (reg1.logll - reg0.logll)
    except AttributeError:
        raise Exception, "Missing or improper log-likelihoods in regression objects"
    if likr < 0.0:  # always enforces positive likelihood ratio
        likr = -likr
    pvalue = stats.chisqprob(likr, 1)
    likratio = {"likr": likr, "df": 1, "p-value": pvalue}
    return likratio


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
