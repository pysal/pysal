"""
Ordinary Least Squares regression with regimes.
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, Daniel Arribas-Bel darribas@asu.edu"

import regimes as REGI
import user_output as USER
import multiprocessing as mp
from ols import BaseOLS
from utils import set_warn, spbroadcast, RegressionProps_basic, RegressionPropsY, spdot
from robust import hac_multi
import summary_output as SUMMARY
import numpy as np
from platform import system
import scipy.sparse as SP


class OLS_Regimes(BaseOLS, REGI.Regimes_Frame, RegressionPropsY):

    """
    Ordinary least squares with results and diagnostics.

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
    w            : pysal W object
                   Spatial weights object (required if running spatial
                   diagnostics)
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None. 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    nonspat_diag : boolean
                   If True, then compute non-spatial diagnostics on
                   the regression.
    spat_diag    : boolean
                   If True, then compute Lagrange multiplier tests (requires
                   w). Note: see moran for further tests.
    moran        : boolean
                   If True, compute Moran's I on the residuals. Note:
                   requires spat_diag=True.
    white_test   : boolean
                   If True, compute White's specification robust test.
                   (requires nonspat_diag=True)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
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
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output


    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    robust       : string
                   Adjustment for robust standard errors
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)                  
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    r2           : float
                   R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    ar2          : float
                   Adjusted R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2ML       : float
                   Sigma squared (maximum likelihood)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    f_stat       : tuple
                   Statistic (float), p-value (float)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   Log likelihood
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    aic          : float
                   Akaike information criterion 
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    schwarz      : float
                   Schwarz information criterion     
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas    
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    t_stat       : list of tuples
                   t statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mulColli     : float
                   Multicollinearity condition number
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    jarque_bera  : dictionary
                   'jb': Jarque-Bera statistic (float); 'pvalue': p-value
                   (float); 'df': degrees of freedom (int)  
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    breusch_pagan : dictionary
                    'bp': Breusch-Pagan statistic (float); 'pvalue': p-value
                    (float); 'df': degrees of freedom (int)  
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    koenker_bassett : dictionary
                      'kb': Koenker-Bassett statistic (float); 'pvalue':
                      p-value (float); 'df': degrees of freedom (int)  
                      Only available in dictionary 'multi' when multiple regressions
                      (see 'multi' below for details)
    white         : dictionary
                    'wh': White statistic (float); 'pvalue': p-value (float);
                    'df': degrees of freedom (int)  
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    lm_error      : tuple
                    Lagrange multiplier test for spatial error model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    lm_lag        : tuple
                    Lagrange multiplier test for spatial lag model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    rlm_error     : tuple
                    Robust lagrange multiplier test for spatial error model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    rlm_lag       : tuple
                    Robust lagrange multiplier test for spatial lag model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    lm_sarma      : tuple
                    Lagrange multiplier test for spatial SARMA model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    moran_res     : tuple
                    Moran's I for the residuals; tuple containing the triple
                    (Moran's I, standardized Moran's I, p-value)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_w        : string
                    Name of weights matrix for use in output
    name_gwk      : string
                    Name of kernel weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    title         : string
                    Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    xtx          : float
                   X'X
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    xtxi         : float
                   (X'X)^-1
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
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
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
    --------
    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it
    the dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = db.by_col(y_var)
    >>> y = np.array(y).reshape(len(y), 1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    We can now run the regression and then have a summary of the output
    by typing: olsr.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> olsr = OLS_Regimes(y, x, regimes, nonspat_diag=False, name_y=y_var, name_x=['PS90','UE90'], name_regimes=r_var, name_ds='NAT')
    >>> olsr.betas
    array([[ 0.39642899],
           [ 0.65583299],
           [ 0.48703937],
           [ 5.59835   ],
           [ 1.16210453],
           [ 0.53163886]])
    >>> np.sqrt(olsr.vm.diagonal())
    array([ 0.24816345,  0.09662678,  0.03628629,  0.46894564,  0.21667395,
            0.05945651])
    >>> olsr.cols2regi
    'all'
    """

    def __init__(self, y, x, regimes,
                 w=None, robust=None, gwk=None, sig2n_k=True,
                 nonspat_diag=True, spat_diag=False, moran=False, white_test=False,
                 vm=False, constant_regi='many', cols2regi='all',
                 regime_err_sep=True, cores=False,
                 name_y=None, name_x=None, name_regimes=None,
                 name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        self.name_x_r = USER.set_name_x(name_x, x)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, add_cons=False)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        if regime_err_sep == True and robust == 'hac':
            set_warn(
                self, "Error by regimes is incompatible with HAC estimation. Hence, error by regimes has been disabled for this model.")
            regime_err_sep = False
        self.regime_err_sep = regime_err_sep
        if regime_err_sep == True and set(cols2regi) == set([True]) and constant_regi == 'many':
            self.y = y
            name_x = USER.set_name_x(name_x, x)
            regi_ids = dict(
                (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
            self._ols_regimes_multi(x, w, regi_ids, cores,
                                    gwk, sig2n_k, robust, nonspat_diag, spat_diag, vm, name_x, moran, white_test)
        else:
            name_x = USER.set_name_x(name_x, x, constant=True)
            x, self.name_x = REGI.Regimes_Frame.__init__(self, x,
                                                         regimes, constant_regi, cols2regi, name_x)
            BaseOLS.__init__(
                self, y=y, x=x, robust=robust, gwk=gwk, sig2n_k=sig2n_k)
            if regime_err_sep == True and robust == None:
                y2, x2 = REGI._get_weighted_var(
                    regimes, self.regimes_set, sig2n_k, self.u, y, x)
                ols2 = BaseOLS(y=y2, x=x2, sig2n_k=sig2n_k)
                RegressionProps_basic(self, betas=ols2.betas, vm=ols2.vm)
                self.title = "ORDINARY LEAST SQUARES - REGIMES (Group-wise heteroskedasticity)"
                nonspat_diag = None
                set_warn(
                    self, "Residuals treated as homoskedastic for the purpose of diagnostics.")
            else:
                self.title = "ORDINARY LEAST SQUARES - REGIMES"
            self.robust = USER.set_robust(robust)
            self.chow = REGI.Chow(self)
            SUMMARY.OLS(reg=self, vm=vm, w=w, nonspat_diag=nonspat_diag,
                        spat_diag=spat_diag, moran=moran, white_test=white_test, regimes=True)

    def _ols_regimes_multi(self, x, w, regi_ids, cores,
                           gwk, sig2n_k, robust, nonspat_diag, spat_diag, vm, name_x, moran, white_test):
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(self.y,x,w,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work,args=(self.y,x,w,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work, args=(
                    self.y, x, w, regi_ids, r, robust, sig2n_k, self.name_ds, self.name_y, name_x, self.name_w, self.name_regimes))
            else:
                results_p[r] = _work(*(self.y, x, w, regi_ids, r, robust, sig2n_k,
                                       self.name_ds, self.name_y, name_x, self.name_w, self.name_regimes))

        self.kryd = 0
        self.kr = x.shape[1] + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
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
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.multi = results
        self.hac_var = x
        if robust == 'hac':
            hac_multi(self, gwk)
        self.chow = REGI.Chow(self)
        if spat_diag:
            self._get_spat_diag_props(x, sig2n_k)
        SUMMARY.OLS_multi(reg=self, multireg=self.multi, vm=vm, nonspat_diag=nonspat_diag,
                          spat_diag=spat_diag, moran=moran, white_test=white_test, regimes=True, w=w)

    def _get_spat_diag_props(self, x, sig2n_k):
        self.k = self.kr
        self._cache = {}
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.x = REGI.regimeX_setup(
            x, self.regimes, [True] * x.shape[1], self.regimes_set)
        self.xtx = spdot(self.x.T, self.x)
        self.xtxi = np.linalg.inv(self.xtx)


def _work(y, x, w, regi_ids, r, robust, sig2n_k, name_ds, name_y, name_x, name_w, name_regimes):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    if robust == 'hac':
        robust = None
    model = BaseOLS(y_r, x_constant, robust=robust, sig2n_k=sig2n_k)
    model.title = "ORDINARY LEAST SQUARES ESTIMATION - REGIME %s" % r
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    if w:
        w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
        set_warn(model, warn)
        model.w = w_r
    return model


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db = pysal.open(pysal.examples.get_path('columbus.dbf'), 'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ['INC', 'HOVAL']
    x = np.array([db.by_col(name) for name in x_var]).T
    r_var = 'NSA'
    regimes = db.by_col(r_var)
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    olsr = OLS_Regimes(y, x, regimes, w=w, constant_regi='many', nonspat_diag=False, spat_diag=False, name_y=y_var, name_x=['INC', 'HOVAL'],
                       name_ds='columbus', name_regimes=r_var, name_w='columbus.gal', regime_err_sep=True, cols2regi=[True, True], sig2n_k=True, robust='white')
    print olsr.summary
