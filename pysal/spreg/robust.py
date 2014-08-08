__author__ = "Luc Anselin luc.anselin@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu, \
        David C. Folch david.folch@asu.edu"

import numpy as np
import numpy.linalg as la
from pysal import lag_spatial
from utils import spdot, spbroadcast
from user_output import check_constant


def robust_vm(reg, gwk=None, sig2n_k=False):
    """
    Robust estimation of the variance-covariance matrix. Estimated by White (default) or HAC (if wk is provided). 

    Parameters
    ----------

    reg             : Regression object (OLS or TSLS)
                      output instance from a regression model

    gwk             : PySAL weights object
                      Optional. Spatial weights based on kernel functions
                      If provided, returns the HAC variance estimation
    sig2n_k         : boolean
                      If True, then use n-k to rescale the vc matrix.
                      If False, use n. (White only)

    Returns
    --------

    psi             : kxk array
                      Robust estimation of the variance-covariance

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from ols import OLS
    >>> from twosls import TSLS
    >>> db=pysal.open(pysal.examples.get_path("NAT.dbf"),"r")
    >>> y = np.array(db.by_col("HR90"))
    >>> y = np.reshape(y, (y.shape[0],1))
    >>> X = []
    >>> X.append(db.by_col("RD90"))
    >>> X.append(db.by_col("DV90"))
    >>> X = np.array(X).T                       

    Example with OLS with unadjusted standard errors

    >>> ols = OLS(y,X)
    >>> ols.vm
    array([[ 0.17004545,  0.00226532, -0.02243898],
           [ 0.00226532,  0.00941319, -0.00031638],
           [-0.02243898, -0.00031638,  0.00313386]])

    Example with OLS and White

    >>> ols = OLS(y,X, robust='white')
    >>> ols.vm
    array([[ 0.24515481,  0.01093322, -0.03441966],
           [ 0.01093322,  0.01798616, -0.00071414],
           [-0.03441966, -0.00071414,  0.0050153 ]])

    Example with OLS and HAC

    >>> wk = pysal.kernelW_from_shapefile(pysal.examples.get_path('NAT.shp'),k=15,function='triangular', fixed=False)
    >>> wk.transform = 'o'
    >>> ols = OLS(y,X, robust='hac', gwk=wk)
    >>> ols.vm
    array([[ 0.29213532,  0.01670361, -0.03948199],
           [ 0.01655557,  0.02295829, -0.00116874],
           [-0.03941483, -0.00119077,  0.00568314]])

    Example with 2SLS and White

    >>> yd = []
    >>> yd.append(db.by_col("UE90"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("UE80"))
    >>> q = np.array(q).T
    >>> tsls = TSLS(y, X, yd, q=q, robust='white')
    >>> tsls.vm
    array([[ 0.29569954,  0.04119843, -0.02496858, -0.01640185],
           [ 0.04119843,  0.03647762,  0.004702  , -0.00987345],
           [-0.02496858,  0.004702  ,  0.00648262, -0.00292891],
           [-0.01640185, -0.00987345, -0.00292891,  0.0053322 ]])

    Example with 2SLS and HAC

    >>> tsls = TSLS(y, X, yd, q=q, robust='hac', gwk=wk)
    >>> tsls.vm
    array([[ 0.41985329,  0.06823119, -0.02883889, -0.02788116],
           [ 0.06867042,  0.04887508,  0.00497443, -0.01367746],
           [-0.02856454,  0.00501402,  0.0072195 , -0.00321604],
           [-0.02810131, -0.01364908, -0.00318197,  0.00713251]])

    """
    if hasattr(reg, 'h'):  # If reg has H, do 2SLS estimator. OLS otherwise.
        tsls = True
        xu = spbroadcast(reg.h, reg.u)
    else:
        tsls = False
        xu = spbroadcast(reg.x, reg.u)

    if gwk:  # If gwk do HAC. White otherwise.
        gwkxu = lag_spatial(gwk, xu)
        psi0 = spdot(xu.T, gwkxu)
    else:
        psi0 = spdot(xu.T, xu)
        if sig2n_k:
            psi0 = psi0 * (1. * reg.n / (reg.n - reg.k))
    if tsls:
        psi1 = spdot(reg.varb, reg.zthhthi)
        psi = spdot(psi1, np.dot(psi0, psi1.T))
    else:
        psi = spdot(reg.xtxi, np.dot(psi0, reg.xtxi))

    return psi


def hac_multi(reg, gwk, constant=False):
    """
    HAC robust estimation of the variance-covariance matrix for multi-regression object 

    Parameters
    ----------

    reg             : Regression object (OLS or TSLS)
                      output instance from a regression model

    gwk             : PySAL weights object
                      Spatial weights based on kernel functions

    Returns
    --------

    psi             : kxk array
                      Robust estimation of the variance-covariance

    """
    if not constant:
        reg.hac_var = check_constant(reg.hac_var)
    xu = spbroadcast(reg.hac_var, reg.u)
    gwkxu = lag_spatial(gwk, xu)
    psi0 = spdot(xu.T, gwkxu)
    counter = 0
    for m in reg.multi:
        reg.multi[m].robust = 'hac'
        reg.multi[m].name_gwk = reg.name_gwk
        try:
            psi1 = spdot(reg.multi[m].varb, reg.multi[m].zthhthi)
            reg.multi[m].vm = spdot(psi1, np.dot(psi0, psi1.T))
        except:
            reg.multi[m].vm = spdot(
                reg.multi[m].xtxi, np.dot(psi0, reg.multi[m].xtxi))
        reg.vm[(counter * reg.kr):((counter + 1) * reg.kr),
               (counter * reg.kr):((counter + 1) * reg.kr)] = reg.multi[m].vm
        counter += 1


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
