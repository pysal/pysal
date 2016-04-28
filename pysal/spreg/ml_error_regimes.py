"""
ML Estimation of Spatial Error Model
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import pysal
import numpy as np
import multiprocessing as mp
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY
import diagnostics as DIAG
from utils import set_warn
from ml_error import BaseML_Error
from platform import system

__all__ = ["ML_Error_Regimes"]


class ML_Error_Regimes(BaseML_Error, REGI.Regimes_Frame):

    """
    ML estimation of the spatial error model with regimes (note no consistency 
    checks, diagnostics or constants added); Anselin (1988) [Anselin1988]_

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: ['one', 'many']
                   Switcher controlling the constant term setup. It may take
                   the following values:
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    w            : Sparse matrix
                   Spatial weights sparse matrix 
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue computation
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    regime_err_sep : boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep : boolean
                   Always False, kept for consistency in function call, ignored.
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    spat_diag    : boolean
                   if True, include spatial diagnostics
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output

    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   (k+1)x1 array of estimated coefficients (lambda last)
    lam          : float
                   estimate of spatial autoregressive coefficient
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
                   if 'ord', Ord eigenvalue computation
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   variance covariance matrix for lambda, sigma (2 x 2)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   maximized log-likelihood (including constant terms)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas    
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: ['one', 'many']
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:
                     *  'one': a vector of ones is appended to x and held
                               constant across regimes
                     * 'many': a vector of ones is appended to x and considered
                               different per regime
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_lag_sep   : boolean
                   If True, the spatial parameter for spatial lag is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression

    Examples
    ________

    Open data baltim.dbf using pysal and create the variables matrices and weights matrix.

    >>> import numpy as np
    >>> import pysal as ps
    >>> db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","AGE","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w_name = "baltim_q.gal"
    >>> w.transform = 'r'    

    Since in this example we are interested in checking whether the results vary
    by regimes, we use CITCOU to define whether the location is in the city or 
    outside the city (in the county):

    >>> regimes = db.by_col("CITCOU")

    Now we can run the regression with all parameters:

    >>> mlerr = ML_Error_Regimes(y,x,regimes,w=w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name,name_regimes="CITCOU")
    >>> np.around(mlerr.betas, decimals=4)
    array([[ -2.3949],
           [  4.8738],
           [ -0.0291],
           [  0.3328],
           [ 31.7962],
           [  2.981 ],
           [ -0.2371],
           [  0.8058],
           [  0.6177]])
    >>> "{0:.6f}".format(mlerr.lam)
    '0.617707'
    >>> "{0:.6f}".format(mlerr.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mlerr.std_y)
    '23.606077'
    >>> np.around(mlerr.vm1, decimals=4)
    array([[   0.005 ,   -0.3535],
           [  -0.3535,  441.3039]])
    >>> np.around(np.diag(mlerr.vm), decimals=4)
    array([ 58.5055,   2.4295,   0.0072,   0.0639,  80.5925,   3.161 ,
             0.012 ,   0.0499,   0.005 ])
    >>> np.around(mlerr.sig2, decimals=4)
    array([[ 209.6064]])
    >>> "{0:.6f}".format(mlerr.logll)
    '-870.333106'
    >>> "{0:.6f}".format(mlerr.aic)
    '1756.666212'
    >>> "{0:.6f}".format(mlerr.schwarz)
    '1783.481077'
    >>> mlerr.title
    'MAXIMUM LIKELIHOOD SPATIAL ERROR - REGIMES (METHOD = full)'
    """

    def __init__(self, y, x, regimes, w=None, constant_regi='many',
                 cols2regi='all', method='full', epsilon=0.0000001,
                 regime_err_sep=False, regime_lag_sep=False, cores=False, spat_diag=False,
                 vm=False, name_y=None, name_x=None,
                 name_w=None, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.regime_err_sep = regime_err_sep
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n
        self.y = y

        x_constant = USER.check_constant(x)
        name_x = USER.set_name_x(name_x, x)
        self.name_x_r = name_x

        cols2regi = REGI.check_cols2regi(constant_regi, cols2regi, x)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        self.regime_err_sep = regime_err_sep

        if regime_err_sep == True:
            if set(cols2regi) == set([True]):
                self._error_regimes_multi(y, x, regimes, w, cores,
                                          method, epsilon, cols2regi, vm, name_x, spat_diag)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_err_sep = True."
        else:
            regimes_att = {}
            regimes_att['x'] = x_constant
            regimes_att['regimes'] = regimes
            regimes_att['cols2regi'] = cols2regi
            x, name_x = REGI.Regimes_Frame.__init__(self, x_constant,
                                                    regimes, constant_regi=None, cols2regi=cols2regi,
                                                    names=name_x)

            BaseML_Error.__init__(
                self, y=y, x=x, w=w, method=method, epsilon=epsilon, regimes_att=regimes_att)

            self.title = "MAXIMUM LIKELIHOOD SPATIAL ERROR - REGIMES" + \
                " (METHOD = " + method + ")"
            self.name_x = USER.set_name_x(name_x, x, constant=True)
            self.name_x.append('lambda')
            self.kf += 1  # Adding a fixed k to account for lambda.
            self.chow = REGI.Chow(self)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            SUMMARY.ML_Error(
                reg=self, w=w, vm=vm, spat_diag=spat_diag, regimes=True)

    def _error_regimes_multi(self, y, x, regimes, w, cores,
                             method, epsilon, cols2regi, vm, name_x, spat_diag):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work_error(*(y,x,regi_ids,r,w,method,epsilon,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work_error,args=(y,x,regi_ids,r,w,method,epsilon,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes, ))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work_error, args=(
                    y, x, regi_ids, r, w, method, epsilon, self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes, ))
            else:
                results_p[r] = _work_error(
                    *(y, x, regi_ids, r, w, method, epsilon, self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes))

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
        self.name_y, self.name_x = [], []
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()

        results = {}
        counter = 0
        for r in self.regimes_set:
            """
            if is_win:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            """
            if not cores:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()

            self.vm[(counter * self.kr):((counter + 1) * self.kr),
                    (counter * self.kr):((counter + 1) * self.kr)] = results[r].vm
            self.betas[
                (counter * self.kr):((counter + 1) * self.kr), ] = results[r].betas
            self.u[regi_ids[r], ] = results[r].u
            self.predy[regi_ids[r], ] = results[r].predy
            self.e_filtered[regi_ids[r], ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        SUMMARY.ML_Error_multi(
            reg=self, multireg=self.multi, vm=vm, spat_diag=spat_diag, regimes=True, w=w)


def _work_error(y, x, regi_ids, r, w, method, epsilon, name_ds, name_y, name_x, name_w, name_regimes):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    model = BaseML_Error(
        y=y_r, x=x_constant, w=w_r, method=method, epsilon=epsilon)
    set_warn(model, warn)
    model.w = w_r
    model.title = "MAXIMUM LIKELIHOOD SPATIAL ERROR - REGIME " + \
        str(r) + " (METHOD = " + method + ")"
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    model.aic = DIAG.akaike(reg=model)
    model.schwarz = DIAG.schwarz(reg=model)
    return model


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == "__main__":
    _test()
    import numpy as np
    import pysal as ps

    db = ps.open(ps.examples.get_path("baltim.dbf"), 'r')
    ds_name = "baltim.dbf"
    y_name = "PRICE"
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y), 1)
    x_names = ["NROOM", "NBATH", "PATIO", "FIREPL",
               "AC", "GAR", "AGE", "LOTSZ", "SQFT"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    w = ww.read()
    ww.close()
    w_name = "baltim_q.gal"
    w.transform = 'r'

    regimes = []
    y_coord = np.array(db.by_col("Y"))
    for i in y_coord:
        if i > 544.5:
            regimes.append("North")
        else:
            regimes.append("South")

    mlerror = ML_Error_Regimes(y, x, regimes, w=w, method='full', name_y=y_name,
                               name_x=x_names, name_w=w_name, name_ds=ds_name, regime_err_sep=False,
                               name_regimes="North")
    print mlerror.summary
