'''
Spatial Two Stages Least Squares with Regimes
'''

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, David C. Folch david.folch@asu.edu"

import numpy as np
import pysal
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY
import multiprocessing as mp
from twosls_regimes import TSLS_Regimes, _optimal_weight
from twosls import BaseTSLS
from utils import set_endog, set_endog_sparse, sp_att, set_warn, sphstack, spdot
from robust import hac_multi
from platform import system


class GM_Lag_Regimes(TSLS_Regimes, REGI.Regimes_Frame):

    """
    Spatial two stage least squares (S2SLS) with regimes; 
    Anselin (1988) [Anselin1988]_

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
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x); cannot be
                   used in combination with h
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
    w            : pysal W object
                   Spatial weights object 
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
    regime_lag_sep: boolean
                   If True (default), the spatial parameter for spatial lag is also
                   computed according to different regimes. If False, 
                   the spatial parameter is fixed accross regimes.
                   Option valid only when regime_err_sep=True
    regime_err_sep: boolean
                   If True, a separate regression is run for each regime.
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.
                   If 'hac', then a HAC consistent estimator of the 
                   variance-covariance matrix is given.
                   If 'ogmm', then Optimal GMM is used to estimate
                   betas and the variance-covariance matrix.
                   Default set to None. 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
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
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_pred       : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    kstar        : integer
                   Number of endogenous variables. 
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    h            : array
                   nxl array of instruments (combination of x and q)
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
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
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
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in 
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    hthi         : float
                   (H'H)^-1
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    zthhthi      : array
                   Z'H(H'H)^-1
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pfora1a2     : array
                   n(zthhthi)'varb
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

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("NAT.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

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

    Since we want to run a spatial lag model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or 
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    This class runs a lag model, which means that includes the spatial lag of
    the dependent variable on the right-hand side of the equation. If we want
    to have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model=GM_Lag_Regimes(y, x, regimes, w=w, regime_lag_sep=False, regime_err_sep=False, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> model.betas
    array([[ 1.28897623],
           [ 0.79777722],
           [ 0.56366891],
           [ 8.73327838],
           [ 1.30433406],
           [ 0.62418643],
           [-0.39993716]])

    Once the model is run, we can have a summary of the output by typing:
    model.summary . Alternatively, we can obtain the standard error of 
    the coefficient estimates by calling:

    >>> model.std_err
    array([ 0.44682888,  0.14358192,  0.05655124,  1.06044865,  0.20184548,
            0.06118262,  0.12387232])

    In the example above, all coefficients but the spatial lag vary
    according to the regime. It is also possible to have the spatial lag
    varying according to the regime, which effective will result in an
    independent spatial lag model estimated for each regime. To run these
    models, the argument regime_lag_sep must be set to True:

    >>> model=GM_Lag_Regimes(y, x, regimes, w=w, regime_lag_sep=True, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> print np.hstack((np.array(model.name_z).reshape(8,1),model.betas,np.sqrt(model.vm.diagonal().reshape(8,1))))
    [['0_CONSTANT' '1.36584769' '0.39854720']
     ['0_PS90' '0.80875730' '0.11324884']
     ['0_UE90' '0.56946813' '0.04625087']
     ['0_W_HR90' '-0.4342438' '0.13350159']
     ['1_CONSTANT' '7.90731073' '1.63601874']
     ['1_PS90' '1.27465703' '0.24709870']
     ['1_UE90' '0.60167693' '0.07993322']
     ['1_W_HR90' '-0.2960338' '0.19934459']]

    Alternatively, we can type: 'model.summary' to see the organized results output.
    The class is flexible enough to accomodate a spatial lag model that,
    besides the spatial lag of the dependent variable, includes other
    non-spatial endogenous regressors. As an example, we will add the endogenous
    variable RD90 (resource deprivation) and we decide to instrument for it with
    FP89 (families below poverty):

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And we can run the model again:

    >>> model = GM_Lag_Regimes(y, x, regimes, yend=yd, q=q, w=w, regime_lag_sep=False, regime_err_sep=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')
    >>> model.betas
    array([[ 3.42195202],
           [ 1.03311878],
           [ 0.14308741],
           [ 8.99740066],
           [ 1.91877758],
           [-0.32084816],
           [ 2.38918212],
           [ 3.67243761],
           [ 0.06959139]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates. Alternatively, we can have a summary of the output by typing:
    model.summary

    >>> model.std_err
    array([ 0.49163311,  0.12237382,  0.05633464,  0.72555909,  0.17250521,
            0.06749131,  0.27370369,  0.25106224,  0.05804213])
    """

    def __init__(self, y, x, regimes, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True,
                 robust=None, gwk=None, sig2n_k=False,
                 spat_diag=False, constant_regi='many',
                 cols2regi='all', regime_lag_sep=False, regime_err_sep=True,
                 cores=False, vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None, name_regimes=None,
                 name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        name_x = USER.set_name_x(name_x, x, constant=True)
        name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        name_q.extend(
            USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.constant_regi = constant_regi
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, yend=yend, add_cons=False)
        self.cols2regi = cols2regi
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        if regime_err_sep == True and robust == 'hac':
            set_warn(
                self, "Error by regimes is incompatible with HAC estimation for Spatial Lag models. Hence, error and lag by regimes have been disabled for this model.")
            regime_err_sep = False
            regime_lag_sep = False
        self.regime_err_sep = regime_err_sep
        self.regime_lag_sep = regime_lag_sep
        if regime_lag_sep == True:
            if not regime_err_sep:
                raise Exception, "regime_err_sep must be True when regime_lag_sep=True."
            cols2regi += [True]
            w_i, regi_ids, warn = REGI.w_regimes(
                w, regimes, self.regimes_set, transform=True, get_ids=True, min_n=len(cols2regi) + 1)
            set_warn(self, warn)

        else:
            cols2regi += [False]

        if regime_err_sep == True and set(cols2regi) == set([True]) and constant_regi == 'many':
            self.y = y
            self.GM_Lag_Regimes_Multi(y, x, w_i, w, regi_ids,
                                      yend=yend, q=q, w_lags=w_lags, lag_q=lag_q, cores=cores,
                                      robust=robust, gwk=gwk, sig2n_k=sig2n_k, cols2regi=cols2regi,
                                      spat_diag=spat_diag, vm=vm, name_y=name_y, name_x=name_x,
                                      name_yend=name_yend, name_q=name_q, name_regimes=self.name_regimes,
                                      name_w=name_w, name_gwk=name_gwk, name_ds=name_ds)
        else:
            if regime_lag_sep == True:
                w = REGI.w_regimes_union(w, w_i, self.regimes_set)
            yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
            name_yend.append(USER.set_name_yend_sp(name_y))
            TSLS_Regimes.__init__(self, y=y, x=x, yend=yend2, q=q2,
                                  regimes=regimes, w=w, robust=robust, gwk=gwk,
                                  sig2n_k=sig2n_k, spat_diag=spat_diag, vm=vm,
                                  constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                  name_y=name_y, name_x=name_x, name_yend=name_yend, name_q=name_q,
                                  name_regimes=name_regimes, name_w=name_w, name_gwk=name_gwk,
                                  name_ds=name_ds, summ=False)
            if regime_lag_sep:
                self.sp_att_reg(w_i, regi_ids, yend2[:, -1].reshape(self.n, 1))
            else:
                self.rho = self.betas[-1]
                self.predy_e, self.e_pred, warn = sp_att(w, self.y, self.predy,
                                                         yend2[:, -1].reshape(self.n, 1), self.rho)
                set_warn(self, warn)
            self.regime_lag_sep = regime_lag_sep
            self.title = "SPATIAL " + self.title
            SUMMARY.GM_Lag(
                reg=self, w=w, vm=vm, spat_diag=spat_diag, regimes=True)

    def GM_Lag_Regimes_Multi(self, y, x, w_i, w, regi_ids, cores=False,
                             yend=None, q=None, w_lags=1, lag_q=True,
                             robust=None, gwk=None, sig2n_k=False, cols2regi='all',
                             spat_diag=False, vm=False, name_y=None, name_x=None,
                             name_yend=None, name_q=None, name_regimes=None,
                             name_w=None, name_gwk=None, name_ds=None):
        #        pool = mp.Pool(cores)
        self.name_ds = USER.set_name_ds(name_ds)
        name_x = USER.set_name_x(name_x, x)
        name_yend.append(USER.set_name_yend_sp(name_y))
        self.name_w = USER.set_name_w(name_w, w_i)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        results_p = {}
        """
        for r in self.regimes_set:
            w_r = w_i[r].sparse
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(y,x,regi_ids,r,yend,q,w_r,w_lags,lag_q,robust,sig2n_k,self.name_ds,name_y,name_x,name_yend,name_q,self.name_w,name_regimes))
            else:                
                results_p[r] = pool.apply_async(_work,args=(y,x,regi_ids,r,yend,q,w_r,w_lags,lag_q,robust,sig2n_k,self.name_ds,name_y,name_x,name_yend,name_q,self.name_w,name_regimes, ))
                is_win = False
        """
        for r in self.regimes_set:
            w_r = w_i[r].sparse
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work, args=(
                    y, x, regi_ids, r, yend, q, w_r, w_lags, lag_q, robust, sig2n_k, self.name_ds, name_y, name_x, name_yend, name_q, self.name_w, name_regimes, ))
            else:
                results_p[r] = _work(*(y, x, regi_ids, r, yend, q, w_r, w_lags, lag_q, robust,
                                       sig2n_k, self.name_ds, name_y, name_x, name_yend, name_q, self.name_w, name_regimes))

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.name_x_r = name_x + name_yend
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
        self.name_y, self.name_x, self.name_yend, self.name_q, self.name_z, self.name_h = [
        ], [], [], [], [], []
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
            results[r].predy_e, results[r].e_pred, warn = sp_att(w_i[r], results[r].y, results[
                                                                 r].predy, results[r].yend[:, -1].reshape(results[r].n, 1), results[r].rho)
            set_warn(results[r], warn)
            results[r].w = w_i[r]
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
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            if r == self.regimes_set[0]:
                self.hac_var = np.zeros((self.n, results[r].h.shape[1]), float)
            self.hac_var[regi_ids[r], ] = results[r].h
            counter += 1
        self.multi = results
        if robust == 'hac':
            hac_multi(self, gwk, constant=True)
        if robust == 'ogmm':
            set_warn(
                self, "Residuals treated as homoskedastic for the purpose of diagnostics.")
        self.chow = REGI.Chow(self)
        if spat_diag:
            pass
            #self._get_spat_diag_props(y, x, w, yend, q, w_lags, lag_q)
        SUMMARY.GM_Lag_multi(
            reg=self, multireg=self.multi, vm=vm, spat_diag=spat_diag, regimes=True, w=w)

    def sp_att_reg(self, w_i, regi_ids, wy):
        predy_e_r, e_pred_r = {}, {}
        self.predy_e = np.zeros((self.n, 1), float)
        self.e_pred = np.zeros((self.n, 1), float)
        counter = 1
        for r in self.regimes_set:
            self.rho = self.betas[(self.kr - self.kryd) * self.nr + self.kf - (
                self.yend.shape[1] - self.nr * self.kryd) + self.kryd * counter - 1]
            self.predy_e[regi_ids[r], ], self.e_pred[regi_ids[r], ], warn = sp_att(w_i[r],
                                                                                   self.y[regi_ids[r]], self.predy[
                                                                                       regi_ids[r]],
                                                                                   wy[regi_ids[r]], self.rho)
            counter += 1

    def _get_spat_diag_props(self, y, x, w, yend, q, w_lags, lag_q):
        self._cache = {}
        yend, q = set_endog(y, x, w, yend, q, w_lags, lag_q)
        x = USER.check_constant(x)
        x = REGI.regimeX_setup(
            x, self.regimes, [True] * x.shape[1], self.regimes_set)
        self.z = sphstack(x, REGI.regimeX_setup(
            yend, self.regimes, [True] * (yend.shape[1] - 1) + [False], self.regimes_set))
        self.h = sphstack(
            x, REGI.regimeX_setup(q, self.regimes, [True] * q.shape[1], self.regimes_set))
        hthi = np.linalg.inv(spdot(self.h.T, self.h))
        zth = spdot(self.z.T, self.h)
        self.varb = np.linalg.inv(spdot(spdot(zth, hthi), zth.T))


def _work(y, x, regi_ids, r, yend, q, w_r, w_lags, lag_q, robust, sig2n_k, name_ds, name_y, name_x, name_yend, name_q, name_w, name_regimes):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    if yend is not None:
        yend_r = yend[regi_ids[r]]
    else:
        yend_r = yend
    if q is not None:
        q_r = q[regi_ids[r]]
    else:
        q_r = q
    yend_r, q_r = set_endog_sparse(y_r, x_r, w_r, yend_r, q_r, w_lags, lag_q)
    x_constant = USER.check_constant(x_r)
    if robust == 'hac' or robust == 'ogmm':
        robust2 = None
    else:
        robust2 = robust
    model = BaseTSLS(
        y_r, x_constant, yend_r, q_r, robust=robust2, sig2n_k=sig2n_k)
    model.title = "SPATIAL TWO STAGE LEAST SQUARES ESTIMATION - REGIME %s" % r
    if robust == 'ogmm':
        _optimal_weight(model, sig2n_k, warn=False)
    model.rho = model.betas[-1]
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_yend = ['%s_%s' % (str(r), i) for i in name_yend]
    model.name_z = model.name_x + model.name_yend
    model.name_q = ['%s_%s' % (str(r), i) for i in name_q]
    model.name_h = model.name_x + model.name_q
    model.name_w = name_w
    model.name_regimes = name_regimes
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
    db = pysal.open(pysal.examples.get_path("columbus.dbf"), 'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ['INC']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ['HOVAL']
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ['DISCBD']
    q = np.array([db.by_col(name) for name in q_var]).T
    r_var = 'NSA'
    regimes = db.by_col(r_var)
    w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    model = GM_Lag_Regimes(y, x, regimes, yend=yd, q=q, w=w, constant_regi='many', spat_diag=True, sig2n_k=False, lag_q=True, name_y=y_var,
                           name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='columbus', name_w='columbus.gal', regime_err_sep=True, robust='white')
    print model.summary
