"""
Spatial Error Models with regimes module
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import multiprocessing as mp
import regimes as REGI
import user_output as USER
import summary_output as SUMMARY
from pysal import lag_spatial
from ols import BaseOLS
from twosls import BaseTSLS
from error_sp import BaseGM_Error, BaseGM_Endog_Error, _momentsGM_Error
from utils import set_endog, iter_msg, sp_att, set_warn
from utils import optim_moments, get_spFilter, get_lags
from utils import spdot, RegressionPropsY
from platform import system


class GM_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    """
    GMM method for a spatial error model with regimes, with results and diagnostics;
    based on Kelejian and Prucha (1998, 1999) [Kelejian1998]_ [Kelejian1999]_.

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
                   Spatial weights object   
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
    regime_err_sep : boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep : boolean
                   Always False, kept for consistency, ignored.
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
    name_w       : string
                   Name of weights matrix for use in output
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
    e_filtered   : array
                   nx1 array of spatially filtered residuals
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
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    vm           : array
                   Variance covariance matrix (kxk)
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
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
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
    regime_err_sep : boolean
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

    >>> import pysal
    >>> import numpy as np

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

    Since we want to run a spatial error model, we need to specify
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

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print model.name_x
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', 'lambda']
    >>> np.around(model.betas, decimals=6)
    array([[ 0.074807],
           [ 0.786107],
           [ 0.538849],
           [ 5.103756],
           [ 1.196009],
           [ 0.600533],
           [ 0.364103]])
    >>> np.around(model.std_err, decimals=6)
    array([ 0.379864,  0.152316,  0.051942,  0.471285,  0.19867 ,  0.057252])
    >>> np.around(model.z_stat, decimals=6)
    array([[  0.196932,   0.843881],
           [  5.161042,   0.      ],
           [ 10.37397 ,   0.      ],
           [ 10.829455,   0.      ],
           [  6.02007 ,   0.      ],
           [ 10.489215,   0.      ]])
    >>> np.around(model.sig2, decimals=6)
    28.172732

    """

    def __init__(self, y, x, regimes, w,
                 vm=False, name_y=None, name_x=None, name_w=None,
                 constant_regi='many', cols2regi='all', regime_err_sep=False,
                 regime_lag_sep=False,
                 cores=False, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
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
                                          cols2regi, vm, name_x)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_err_sep = True."
        else:
            self.x, self.name_x = REGI.Regimes_Frame.__init__(self, x_constant,
                                                              regimes, constant_regi=None, cols2regi=cols2regi, names=name_x)
            ols = BaseOLS(y=y, x=self.x)
            self.k = ols.x.shape[1]
            moments = _momentsGM_Error(w, ols.u)
            lambda1 = optim_moments(moments)
            xs = get_spFilter(w, lambda1, x_constant)
            ys = get_spFilter(w, lambda1, y)
            xs = REGI.Regimes_Frame.__init__(self, xs,
                                             regimes, constant_regi=None, cols2regi=cols2regi)[0]
            ols2 = BaseOLS(y=ys, x=xs)

            # Output
            self.predy = spdot(self.x, ols2.betas)
            self.u = y - self.predy
            self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
            self.sig2 = ols2.sig2n
            self.e_filtered = self.u - lambda1 * lag_spatial(w, self.u)
            self.vm = self.sig2 * ols2.xtxi
            self.title = "SPATIALLY WEIGHTED LEAST SQUARES - REGIMES"
            self.name_x.append('lambda')
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}
            SUMMARY.GM_Error(reg=self, w=w, vm=vm, regimes=True)

    def _error_regimes_multi(self, y, x, regimes, w, cores,
                             cols2regi, vm, name_x):
        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                results_p[r] = _work_error(*(y,x,regi_ids,r,w,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes))
                is_win = True
            else:
                pool = mp.Pool(cores)                
                results_p[r] = pool.apply_async(_work_error,args=(y,x,regi_ids,r,w,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes, ))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work_error, args=(
                    y, x, regi_ids, r, w, self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes, ))
            else:
                results_p[r] = _work_error(
                    *(y, x, regi_ids, r, w, self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes))

        self.kryd = 0
        self.kr = len(cols2regi)
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * (self.kr + 1), 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
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
                (counter * (self.kr + 1)):((counter + 1) * (self.kr + 1)), ] = results[r].betas
            self.u[regi_ids[r], ] = results[r].u
            self.predy[regi_ids[r], ] = results[r].predy
            self.e_filtered[regi_ids[r], ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        SUMMARY.GM_Error_multi(
            reg=self, multireg=self.multi, vm=vm, regimes=True)


class GM_Endog_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    '''
    GMM method for a spatial error model with regimes and endogenous variables, with
    results and diagnostics; based on Kelejian and Prucha (1998,
    1999) [Kelejian1998]_ [Kelejian1999]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object   
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
    regime_err_sep : boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep : boolean
                   Always False, kept for consistency, ignored.
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
    e_filtered   : array
                   nx1 array of spatially filtered residuals
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
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
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
    sig2         : float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regimes variable for use in output
    title         : string
                    Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : ['one', 'many']
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:
                      *  'one': a vector of ones is appended to x and held
                                constant across regimes
                      * 'many': a vector of ones is appended to x and considered
                                different per regime
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_err_sep : boolean
                   If True, a separate regression is run for each regime.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
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

    >>> import pysal
    >>> import numpy as np

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

    For the endogenous models, we add the endogenous variable RD90 (resource deprivation)
    and we decide to instrument for it with FP89 (families below poverty):

    >>> yd_var = ['RD90']
    >>> yend = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already 
    existing gal file or create a new one. In this case, we will create one 
    from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Endog_Error_Regimes(y, x, yend, q, regimes, w=w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included. Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', 'lambda']
    >>> np.around(model.betas, decimals=5)
    array([[ 3.59718],
           [ 1.0652 ],
           [ 0.15822],
           [ 9.19754],
           [ 1.88082],
           [-0.24878],
           [ 2.46161],
           [ 3.57943],
           [ 0.25564]])
    >>> np.around(model.std_err, decimals=6)
    array([ 0.522633,  0.137555,  0.063054,  0.473654,  0.18335 ,  0.072786,
            0.300711,  0.240413])

    '''

    def __init__(self, y, x, yend, q, regimes, w, cores=False,
                 vm=False, constant_regi='many', cols2regi='all',
                 regime_err_sep=False, regime_lag_sep=False, name_y=None,
                 name_x=None, name_yend=None, name_q=None, name_w=None,
                 name_ds=None, name_regimes=None, summ=True, add_lag=False):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_w = USER.set_name_w(name_w, w)
        self.n = n
        self.y = y

        name_x = USER.set_name_x(name_x, x)
        if summ:
            name_yend = USER.set_name_yend(name_yend, yend)
            self.name_y = USER.set_name_y(name_y)
            name_q = USER.set_name_q(name_q, q)
        self.name_x_r = name_x + name_yend

        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, yend=yend)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        self.regime_err_sep = regime_err_sep

        if regime_err_sep == True:
            if set(cols2regi) == set([True]):
                self._endog_error_regimes_multi(y, x, regimes, w, yend, q, cores,
                                                cols2regi, vm, name_x, name_yend, name_q, add_lag)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_err_sep = True."
        else:
            x_constant = USER.check_constant(x)
            q, name_q = REGI.Regimes_Frame.__init__(self, q,
                                                    regimes, constant_regi=None, cols2regi='all', names=name_q)
            x, name_x = REGI.Regimes_Frame.__init__(self, x_constant,
                                                    regimes, constant_regi=None, cols2regi=cols2regi,
                                                    names=name_x)
            yend2, name_yend = REGI.Regimes_Frame.__init__(self, yend,
                                                           regimes, constant_regi=None,
                                                           cols2regi=cols2regi, yend=True, names=name_yend)

            tsls = BaseTSLS(y=y, x=x, yend=yend2, q=q)
            self.k = tsls.z.shape[1]
            self.x = tsls.x
            self.yend, self.z = tsls.yend, tsls.z
            moments = _momentsGM_Error(w, tsls.u)
            lambda1 = optim_moments(moments)
            xs = get_spFilter(w, lambda1, x_constant)
            xs = REGI.Regimes_Frame.__init__(self, xs,
                                             regimes, constant_regi=None, cols2regi=cols2regi)[0]
            ys = get_spFilter(w, lambda1, y)
            yend_s = get_spFilter(w, lambda1, yend)
            yend_s = REGI.Regimes_Frame.__init__(self, yend_s,
                                                 regimes, constant_regi=None, cols2regi=cols2regi,
                                                 yend=True)[0]
            tsls2 = BaseTSLS(ys, xs, yend_s, h=tsls.h)

            # Output
            self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
            self.predy = spdot(tsls.z, tsls2.betas)
            self.u = y - self.predy
            self.sig2 = float(np.dot(tsls2.u.T, tsls2.u)) / self.n
            self.e_filtered = self.u - lambda1 * lag_spatial(w, self.u)
            self.vm = self.sig2 * tsls2.varb
            self.name_x = USER.set_name_x(name_x, x, constant=True)
            self.name_yend = USER.set_name_yend(name_yend, yend)
            self.name_z = self.name_x + self.name_yend
            self.name_z.append('lambda')
            self.name_q = USER.set_name_q(name_q, q)
            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}
            if summ:
                self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES - REGIMES"
                SUMMARY.GM_Endog_Error(reg=self, w=w, vm=vm, regimes=True)

    def _endog_error_regimes_multi(self, y, x, regimes, w, yend, q, cores,
                                   cols2regi, vm, name_x, name_yend, name_q, add_lag):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
        if add_lag != False:
            self.cols2regi += [True]
            cols2regi += [True]
            self.predy_e = np.zeros((self.n, 1), float)
            self.e_pred = np.zeros((self.n, 1), float)
        results_p = {}
        for r in self.regimes_set:
            """
            if system() == 'Windows':
                results_p[r] = _work_endog_error(*(y,x,yend,q,regi_ids,r,w,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag))
                is_win = True
            else:
                pool = mp.Pool(cores)        
                results_p[r] = pool.apply_async(_work_endog_error,args=(y,x,yend,q,regi_ids,r,w,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag, ))
                is_win = False
            """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work_endog_error, args=(
                    y, x, yend, q, regi_ids, r, w, self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes, add_lag, ))
            else:
                results_p[r] = _work_endog_error(
                    *(y, x, yend, q, regi_ids, r, w, self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes, add_lag))

        self.kryd, self.kf = 0, 0
        self.kr = len(cols2regi)
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * (self.kr + 1), 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
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

            self.vm[(counter * self.kr):((counter + 1) * self.kr),
                    (counter * self.kr):((counter + 1) * self.kr)] = results[r].vm
            self.betas[
                (counter * (self.kr + 1)):((counter + 1) * (self.kr + 1)), ] = results[r].betas
            self.u[regi_ids[r], ] = results[r].u
            self.predy[regi_ids[r], ] = results[r].predy
            self.e_filtered[regi_ids[r], ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            if add_lag != False:
                self.predy_e[regi_ids[r], ] = results[r].predy_e
                self.e_pred[regi_ids[r], ] = results[r].e_pred
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        if add_lag != False:
            SUMMARY.GM_Combo_multi(
                reg=self, multireg=self.multi, vm=vm, regimes=True)
        else:
            SUMMARY.GM_Endog_Error_multi(
                reg=self, multireg=self.multi, vm=vm, regimes=True)


class GM_Combo_Regimes(GM_Endog_Error_Regimes, REGI.Regimes_Frame):

    """
    GMM method for a spatial lag and error model with regimes and endogenous
    variables, with results and diagnostics; based on Kelejian and Prucha (1998,
    1999) [Kelejian1998]_ [Kelejian1999]_.

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
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object (always needed)   
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
    regime_err_sep : boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep   : boolean
                   If True, the spatial parameter for spatial lag is also
                   computed according to different regimes. If False (default), 
                   the spatial parameter is fixed accross regimes.
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
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
    e_filtered   : array
                   nx1 array of spatially filtered residuals
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
    z            : array
                   nxk array of variables (combination of x and yend)
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
    sig2         : float
                   Sigma squared used in computations (based on filtered
                   residuals)
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
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in 
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regimes variable for use in output
    title         : string
                    Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : ['one', 'many']
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:
                      *  'one': a vector of ones is appended to x and held
                                constant across regimes
                      * 'many': a vector of ones is appended to x and considered
                                different per regime
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep    : boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default), 
                    the spatial parameter is fixed accross regimes.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
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

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Combo_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can have a summary of the
    output by typing: model.summary 
    Alternatively, we can check the betas:

    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '_Global_W_HR90', 'lambda']
    >>> print np.around(model.betas,4)
    [[ 1.4607]
     [ 0.958 ]
     [ 0.5658]
     [ 9.113 ]
     [ 1.1338]
     [ 0.6517]
     [-0.4583]
     [ 0.6136]]

    And lambda:

    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.6136]

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. In this case we consider RD90 (resource deprivation)
    as an endogenous regressor.  We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GM_Combo_Regimes(y, x, regimes, yd, q, w=w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT')
    >>> print model.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', '_Global_W_HR90', 'lambda']
    >>> print model.betas
    [[ 3.41963782]
     [ 1.04065841]
     [ 0.16634393]
     [ 8.86544628]
     [ 1.85120528]
     [-0.24908469]
     [ 2.43014046]
     [ 3.61645481]
     [ 0.03308671]
     [ 0.18684992]]
    >>> print np.sqrt(model.vm.diagonal())
    [ 0.53067577  0.13271426  0.06058025  0.76406411  0.17969783  0.07167421
      0.28943121  0.25308326  0.06126529]
    >>> print 'lambda: ', np.around(model.betas[-1], 4)
    lambda:  [ 0.1868]
    """

    def __init__(self, y, x, regimes, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True, cores=False,
                 constant_regi='many', cols2regi='all',
                 regime_err_sep=False, regime_lag_sep=False,
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        name_x = USER.set_name_x(name_x, x, constant=True)
        self.name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        name_q.extend(
            USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))

        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, yend=yend, add_cons=False)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, n, x.shape[1])
        self.regime_err_sep = regime_err_sep
        self.regime_lag_sep = regime_lag_sep

        if regime_lag_sep == True:
            if regime_err_sep == False:
                raise Exception, "For spatial combo models, if spatial lag is set by regimes (regime_lag_sep=True), spatial error must also be set by regimes (regime_err_sep=True)."
            add_lag = [w_lags, lag_q]
        else:
            if regime_err_sep == True:
                raise Exception, "For spatial combo models, if spatial error is set by regimes (regime_err_sep=True), all coefficients including lambda (regime_lag_sep=True) must be set by regimes."
            cols2regi += [False]
            add_lag = False
            yend, q = set_endog(y, x, w, yend, q, w_lags, lag_q)
        name_yend.append(USER.set_name_yend_sp(self.name_y))

        GM_Endog_Error_Regimes.__init__(self, y=y, x=x, yend=yend,
                                        q=q, regimes=regimes, w=w, vm=vm, constant_regi=constant_regi,
                                        cols2regi=cols2regi, regime_err_sep=regime_err_sep, cores=cores,
                                        name_y=self.name_y, name_x=name_x,
                                        name_yend=name_yend, name_q=name_q, name_w=name_w,
                                        name_ds=name_ds, name_regimes=name_regimes, summ=False, add_lag=add_lag)

        if regime_err_sep != True:
            self.rho = self.betas[-2]
            self.predy_e, self.e_pred, warn = sp_att(w, self.y,
                                                     self.predy, yend[:, -1].reshape(self.n, 1), self.rho)
            set_warn(self, warn)
            self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES - REGIMES"
            SUMMARY.GM_Combo(reg=self, w=w, vm=vm, regimes=True)


def _work_error(y, x, regi_ids, r, w, name_ds, name_y, name_x, name_w, name_regimes):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    model = BaseGM_Error(y_r, x_constant, w_r.sparse)
    set_warn(model, warn)
    model.w = w_r
    model.title = "SPATIALLY WEIGHTED LEAST SQUARES ESTIMATION - REGIME %s" % r
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    return model


def _work_endog_error(y, x, yend, q, regi_ids, r, w, name_ds, name_y, name_x, name_yend, name_q, name_w, name_regimes, add_lag):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    if yend is not None:
        yend_r = yend[regi_ids[r]]
        q_r = q[regi_ids[r]]
    else:
        yend_r, q_r = None, None
    if add_lag != False:
        yend_r, q_r = set_endog(
            y_r, x_r, w_r, yend_r, q_r, add_lag[0], add_lag[1])
    x_constant = USER.check_constant(x_r)
    model = BaseGM_Endog_Error(y_r, x_constant, yend_r, q_r, w_r.sparse)
    set_warn(model, warn)
    if add_lag != False:
        model.rho = model.betas[-2]
        model.predy_e, model.e_pred, warn = sp_att(w_r, model.y,
                                                   model.predy, model.yend[:, -1].reshape(model.n, 1), model.rho)
        set_warn(model, warn)
    model.w = w_r
    model.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES - REGIME %s" % r
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_yend = ['%s_%s' % (str(r), i) for i in name_yend]
    model.name_z = model.name_x + model.name_yend + ['lambda']
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
    import pysal
    import numpy as np
    dbf = pysal.open(pysal.examples.get_path('columbus.dbf'), 'r')
    y = np.array([dbf.by_col('CRIME')]).T
    names_to_extract = ['INC']
    x = np.array([dbf.by_col(name) for name in names_to_extract]).T
    yd_var = ['HOVAL']
    yend = np.array([dbf.by_col(name) for name in yd_var]).T
    q_var = ['DISCBD']
    q = np.array([dbf.by_col(name) for name in q_var]).T
    regimes = regimes = dbf.by_col('NSA')
    w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    w.transform = 'r'
    model = GM_Error_Regimes(y, x, regimes=regimes, w=w, name_y='crime', name_x=[
                             'income'], name_regimes='nsa', name_ds='columbus', regime_err_sep=True)
    print model.summary
