"""
ML Estimation of Spatial Lag Model with Regimes
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import pysal
import numpy as np
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY
import diagnostics as DIAG
import multiprocessing as mp
from ml_lag import BaseML_Lag
from utils import set_warn
from platform import system

__all__ = ["ML_Lag_Regimes"]


class ML_Lag_Regimes(BaseML_Lag, REGI.Regimes_Frame):

    """
    ML estimation of the spatial lag model with regimes (note no consistency 
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
                   if 'ord', Ord eigenvalue method
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    regime_lag_sep: boolean
                   If True, the spatial parameter for spatial lag is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
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
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    u            : array
                   nx1 array of residuals
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
                   if 'ord', Ord eigenvalue method
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
                   Variance covariance matrix (k+2 x k+2), includes sig2
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
    aic          : float
                   Akaike information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    schwarz      : float
                   Schwarz criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
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
    regime_err_sep  : boolean
                   always set to False - kept for compatibility with other
                   regime models
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

    >>> mllag = ML_Lag_Regimes(y,x,regimes,w=w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name,name_regimes="CITCOU")
    >>> np.around(mllag.betas, decimals=4)
    array([[-15.0059],
           [  4.496 ],
           [ -0.0318],
           [  0.35  ],
           [ -4.5404],
           [  3.9219],
           [ -0.1702],
           [  0.8194],
           [  0.5385]])
    >>> "{0:.6f}".format(mllag.rho)
    '0.538503'
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4)
    array([  47.42  ,    2.3953,    0.0051,    0.0648,   69.6765,    3.2066,
              0.0116,    0.0486,    0.004 ,  390.7274])
    >>> np.around(np.diag(mllag.vm), decimals=4)
    array([ 47.42  ,   2.3953,   0.0051,   0.0648,  69.6765,   3.2066,
             0.0116,   0.0486,   0.004 ])
    >>> "{0:.6f}".format(mllag.sig2)
    '200.044334'
    >>> "{0:.6f}".format(mllag.logll)
    '-864.985056'
    >>> "{0:.6f}".format(mllag.aic)
    '1747.970112'
    >>> "{0:.6f}".format(mllag.schwarz)
    '1778.136835'
    >>> mllag.title
    'MAXIMUM LIKELIHOOD SPATIAL LAG - REGIMES (METHOD = full)'
    """

    def __init__(self, y, x, regimes, w=None, constant_regi='many',
                 cols2regi='all', method='full', epsilon=0.0000001,
                 regime_lag_sep=False, regime_err_sep=False, cores=False, spat_diag=False,
                 vm=False, name_y=None, name_x=None,
                 name_w=None, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        USER.check_spat_diag(spat_diag, w)
        name_y = USER.set_name_y(name_y)
        self.name_y = name_y
        self.name_x_r = USER.set_name_x(
            name_x, x) + [USER.set_name_yend_sp(name_y)]
        self.method = method
        self.epsilon = epsilon
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.constant_regi = constant_regi
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, add_cons=False)
        self.cols2regi = cols2regi
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        self.regime_lag_sep = regime_lag_sep
        self._cache = {}
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])

        # regime_err_sep is ignored, always False

        if regime_lag_sep == True:
            if not (set(cols2regi) == set([True]) and constant_regi == 'many'):
                raise Exception, "All variables must vary by regimes if regime_lag_sep = True."
            cols2regi += [True]
            w_i, regi_ids, warn = REGI.w_regimes(
                w, regimes, self.regimes_set, transform=True, get_ids=True, min_n=len(cols2regi) + 1)
            set_warn(self, warn)
        else:
            cols2regi += [False]

        if set(cols2regi) == set([True]) and constant_regi == 'many':
            self.y = y
            self.ML_Lag_Regimes_Multi(y, x, w_i, w, regi_ids,
                                      cores=cores, cols2regi=cols2regi, method=method, epsilon=epsilon,
                                      spat_diag=spat_diag, vm=vm, name_y=name_y, name_x=name_x,
                                      name_regimes=self.name_regimes,
                                      name_w=name_w, name_ds=name_ds)
        else:
            # if regime_lag_sep == True:
            #    w = REGI.w_regimes_union(w, w_i, self.regimes_set)
            name_x = USER.set_name_x(name_x, x, constant=True)
            x, self.name_x = REGI.Regimes_Frame.__init__(self, x,
                                                         regimes, constant_regi, cols2regi=cols2regi[:-1], names=name_x)
            self.name_x.append("_Global_" + USER.set_name_yend_sp(name_y))
            BaseML_Lag.__init__(
                self, y=y, x=x, w=w, method=method, epsilon=epsilon)
            self.kf += 1  # Adding a fixed k to account for spatial lag in Chow
            # adding a fixed k to account for spatial lag in aic, sc
            self.k += 1
            self.chow = REGI.Chow(self)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            self.regime_lag_sep = regime_lag_sep
            self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG - REGIMES" + \
                " (METHOD = " + method + ")"
            SUMMARY.ML_Lag(
                reg=self, w=w, vm=vm, spat_diag=spat_diag, regimes=True)

    def ML_Lag_Regimes_Multi(self, y, x, w_i, w, regi_ids,
                             cores, cols2regi, method, epsilon,
                             spat_diag, vm, name_y, name_x,
                             name_regimes, name_w, name_ds):
        #        pool = mp.Pool(cores)
        name_x = USER.set_name_x(name_x, x) + [USER.set_name_yend_sp(name_y)]
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(y,x,regi_ids,r,w_i[r],method,epsilon,name_ds,name_y,name_x,name_w,name_regimes))
            else:                
                results_p[r] = pool.apply_async(_work,args=(y,x,regi_ids,r,w_i[r],method,epsilon,name_ds,name_y,name_x,name_w,name_regimes, ))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work, args=(y, x, regi_ids, r, w_i[
                                                r], method, epsilon, name_ds, name_y, name_x, name_w, name_regimes, ))
            else:
                results_p[r] = _work(
                    *(y, x, regi_ids, r, w_i[r], method, epsilon, name_ds, name_y, name_x, name_w, name_regimes))

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.name_x_r = name_x
        self.name_regimes = name_regimes
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.predy_e = np.zeros((self.n, 1), float)
        self.e_pred = np.zeros((self.n, 1), float)
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()

        results = {}
        self.name_y, self.name_x = [], []
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
            self.predy_e[regi_ids[r], ] = results[r].predy_e
            self.e_pred[regi_ids[r], ] = results[r].e_pred
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.multi = results
        self.chow = REGI.Chow(self)
        SUMMARY.ML_Lag_multi(
            reg=self, multireg=self.multi, vm=vm, spat_diag=spat_diag, regimes=True, w=w)


def _work(y, x, regi_ids, r, w_r, method, epsilon, name_ds, name_y, name_x, name_w, name_regimes):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    model = BaseML_Lag(y_r, x_constant, w_r, method=method, epsilon=epsilon)
    model.title = "MAXIMUM LIKELIHOOD SPATIAL LAG - REGIME " + \
        str(r) + " (METHOD = " + method + ")"
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    model.k += 1  # add 1 for proper df and aic, sc
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
    regimes = db.by_col("CITCOU")

    mllag = ML_Lag_Regimes(y, x, regimes, w=w, method='full', name_y=y_name, name_x=x_names,
                           name_w=w_name, name_ds=ds_name, regime_lag_sep=True, constant_regi='many',
                           name_regimes="CITCOU")
    print mllag.summary
