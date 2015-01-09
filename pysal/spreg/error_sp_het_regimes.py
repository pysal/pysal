'''
Spatial Error with Heteroskedasticity and Regimes family of models
'''
__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import multiprocessing as mp
import user_output as USER
import summary_output as SUMMARY
import utils as UTILS
import regimes as REGI
from ols import BaseOLS
from twosls import BaseTSLS
from error_sp_het import BaseGM_Error_Het, BaseGM_Endog_Error_Het, get_psi_sigma, get_vc_het, get_vm_het, get_P_hat, get_a1a2, get_vc_het_tsls, get_Omega_GS2SLS
from utils import RegressionPropsY, spdot, set_endog, sphstack, set_warn, sp_att
from scipy import sparse as SP
from pysal import lag_spatial
from platform import system


class GM_Error_Het_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    """
    GMM method for a spatial error model with heteroskedasticity and regimes;
    based on Arraiz et al [1]_, following Anselin [2]_.

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
    regime_err_sep: boolean
                   If True, a separate regression is run for each regime.
    regime_lag_sep : boolean
                   Always False, kept for consistency, ignored.
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from Arraiz et al. 
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
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    multi         : dictionary
                    Only available when multiple regressions are estimated,
                    i.e. when regime_err_sep=True and no variable is fixed
                    across regimes.
                    Contains all attributes of each individual regression

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    .. [2] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity

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

    >>> reg = GM_Error_Het_Regimes(y, x, regimes, w=w, step1c=True, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``pysal.spreg.error_sp``, it allows for inference on the spatial
    parameter. Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print reg.name_x
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', 'lambda']
    >>> np.around(reg.betas, decimals=6)
    array([[ 0.009121],
           [ 0.812973],
           [ 0.549355],
           [ 5.00279 ],
           [ 1.200929],
           [ 0.614681],
           [ 0.429277]])
    >>> np.around(reg.std_err, decimals=6)
    array([ 0.355844,  0.221743,  0.059276,  0.686764,  0.35843 ,  0.092788,
            0.02524 ])

    """

    def __init__(self, y, x, regimes, w, max_iter=1, epsilon=0.00001, step1c=False,
                 constant_regi='many', cols2regi='all', regime_err_sep=False,
                 regime_lag_sep=False,
                 cores=False, vm=False, name_y=None, name_x=None, name_w=None,
                 name_ds=None, name_regimes=None):

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
        self.n, self.step1c = n, step1c
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
                                          max_iter, epsilon, step1c,
                                          cols2regi, vm, name_x)
            else:
                raise Exception, "All coefficients must vary accross regimes if regime_err_sep = True."
        else:
            self.x, self.name_x = REGI.Regimes_Frame.__init__(self, x_constant,
                                                              regimes, constant_regi=None, cols2regi=cols2regi, names=name_x)
            ols = BaseOLS(y=y, x=self.x)
            self.k = ols.x.shape[1]
            wA1 = UTILS.get_A1_het(w.sparse)

            # 1b. GMM --> \tilde{\lambda1}
            moments = UTILS._moments2eqs(wA1, w.sparse, ols.u)
            lambda1 = UTILS.optim_moments(moments)

            if step1c:
                # 1c. GMM --> \tilde{\lambda2}
                sigma = get_psi_sigma(w.sparse, ols.u, lambda1)
                vc1 = get_vc_het(w.sparse, wA1, sigma)
                lambda2 = UTILS.optim_moments(moments, vc1)
            else:
                lambda2 = lambda1
            lambda_old = lambda2

            self.iteration, eps = 0, 1
            while self.iteration < max_iter and eps > epsilon:
                # 2a. reg -->\hat{betas}
                xs = UTILS.get_spFilter(w, lambda_old, x_constant)
                ys = UTILS.get_spFilter(w, lambda_old, y)
                xs = REGI.Regimes_Frame.__init__(self, xs,
                                                 regimes, constant_regi=None, cols2regi=cols2regi)[0]
                ols_s = BaseOLS(y=ys, x=xs)
                self.predy = spdot(self.x, ols_s.betas)
                self.u = self.y - self.predy

                # 2b. GMM --> \hat{\lambda}
                sigma_i = get_psi_sigma(w.sparse, self.u, lambda_old)
                vc_i = get_vc_het(w.sparse, wA1, sigma_i)
                moments_i = UTILS._moments2eqs(wA1, w.sparse, self.u)
                lambda3 = UTILS.optim_moments(moments_i, vc_i)
                eps = abs(lambda3 - lambda_old)
                lambda_old = lambda3
                self.iteration += 1

            self.iter_stop = UTILS.iter_msg(self.iteration, max_iter)

            sigma = get_psi_sigma(w.sparse, self.u, lambda3)
            vc3 = get_vc_het(w.sparse, wA1, sigma)
            self.vm = get_vm_het(moments_i[0], lambda3, self, w.sparse, vc3)
            self.betas = np.vstack((ols_s.betas, lambda3))
            self.e_filtered = self.u - lambda3 * lag_spatial(w, self.u)
            self.title = "SPATIALLY WEIGHTED LEAST SQUARES (HET) - REGIMES"
            self.name_x.append('lambda')
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}

            SUMMARY.GM_Error_Het(reg=self, w=w, vm=vm, regimes=True)

    def _error_regimes_multi(self, y, x, regimes, w, cores,
                             max_iter, epsilon, step1c, cols2regi, vm, name_x):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work_error(*(y,x,regi_ids,r,w,max_iter,epsilon,step1c,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work_error,args=(y,x,regi_ids,r,w,max_iter,epsilon,step1c,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes, ))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work_error, args=(
                    y, x, regi_ids, r, w, max_iter, epsilon, step1c, self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes, ))
            else:
                results_p[r] = _work_error(*(y, x, regi_ids, r, w, max_iter, epsilon, step1c,
                                             self.name_ds, self.name_y, name_x + ['lambda'], self.name_w, self.name_regimes))

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
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
                (counter * self.kr):((counter + 1) * self.kr), ] = results[r].betas
            self.u[regi_ids[r], ] = results[r].u
            self.predy[regi_ids[r], ] = results[r].predy
            self.e_filtered[regi_ids[r], ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        SUMMARY.GM_Error_Het_multi(
            reg=self, multireg=self.multi, vm=vm, regimes=True)


class GM_Endog_Error_Het_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    """
    GMM method for a spatial error model with heteroskedasticity, regimes and
    endogenous variables, with results and diagnostics; based on Arraiz et al
    [1]_, following Anselin [2]_.

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
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from Arraiz et al. 
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    multi         : dictionary
                    Only available when multiple regressions are estimated,
                    i.e. when regime_err_sep=True and no variable is fixed
                    across regimes.
                    Contains all attributes of each individual regression

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    .. [2] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity

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

    >>> reg = GM_Endog_Error_Het_Regimes(y, x, yend, q, regimes, w=w, step1c=True, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``pysal.spreg.error_sp``, it allows for inference on the spatial
    parameter. Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print reg.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', 'lambda']

    >>> print np.around(reg.betas,4)
    [[ 3.5944]
     [ 1.065 ]
     [ 0.1587]
     [ 9.184 ]
     [ 1.8784]
     [-0.2466]
     [ 2.4617]
     [ 3.5756]
     [ 0.2908]]

    >>> print np.around(np.sqrt(reg.vm.diagonal()),4)
    [ 0.5043  0.2132  0.0581  0.6681  0.3504  0.0999  0.3686  0.3402  0.028 ]

    """

    def __init__(self, y, x, yend, q, regimes, w,
                 max_iter=1, epsilon=0.00001, step1c=False,
                 constant_regi='many', cols2regi='all', regime_err_sep=False,
                 regime_lag_sep=False,
                 inv_method='power_exp', cores=False,
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None, name_w=None, name_ds=None,
                 name_regimes=None, summ=True, add_lag=False):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_w = USER.set_name_w(name_w, w)
        self.n, self.step1c = n, step1c
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
                                                max_iter, epsilon, step1c, inv_method, cols2regi, vm,
                                                name_x, name_yend, name_q, add_lag)
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

            # 1a. S2SLS --> \tilde{\delta}
            tsls = BaseTSLS(y=y, x=x, yend=yend2, q=q)
            self.k = tsls.z.shape[1]
            self.x = tsls.x
            self.yend, self.z, self.h = tsls.yend, tsls.z, tsls.h
            wA1 = UTILS.get_A1_het(w.sparse)

            # 1b. GMM --> \tilde{\lambda1}
            moments = UTILS._moments2eqs(wA1, w.sparse, tsls.u)
            lambda1 = UTILS.optim_moments(moments)

            if step1c:
                # 1c. GMM --> \tilde{\lambda2}
                self.u = tsls.u
                zs = UTILS.get_spFilter(w, lambda1, self.z)
                vc1 = get_vc_het_tsls(
                    w.sparse, wA1, self, lambda1, tsls.pfora1a2, zs, inv_method, filt=False)
                lambda2 = UTILS.optim_moments(moments, vc1)
            else:
                lambda2 = lambda1
            lambda_old = lambda2

            self.iteration, eps = 0, 1
            while self.iteration < max_iter and eps > epsilon:
                # 2a. reg -->\hat{betas}
                xs = UTILS.get_spFilter(w, lambda1, x_constant)
                xs = REGI.Regimes_Frame.__init__(self, xs,
                                                 regimes, constant_regi=None, cols2regi=cols2regi)[0]
                ys = UTILS.get_spFilter(w, lambda1, y)
                yend_s = UTILS.get_spFilter(w, lambda1, yend)
                yend_s = REGI.Regimes_Frame.__init__(self, yend_s,
                                                     regimes, constant_regi=None, cols2regi=cols2regi,
                                                     yend=True)[0]
                tsls_s = BaseTSLS(ys, xs, yend_s, h=tsls.h)
                self.predy = spdot(self.z, tsls_s.betas)
                self.u = self.y - self.predy

                # 2b. GMM --> \hat{\lambda}
                vc2 = get_vc_het_tsls(
                    w.sparse, wA1, self, lambda_old, tsls_s.pfora1a2, sphstack(xs, yend_s), inv_method)
                moments_i = UTILS._moments2eqs(wA1, w.sparse, self.u)
                lambda3 = UTILS.optim_moments(moments_i, vc2)
                eps = abs(lambda3 - lambda_old)
                lambda_old = lambda3
                self.iteration += 1

            self.iter_stop = UTILS.iter_msg(self.iteration, max_iter)

            zs = UTILS.get_spFilter(w, lambda3, self.z)
            P = get_P_hat(self, tsls.hthi, zs)
            vc3 = get_vc_het_tsls(
                w.sparse, wA1, self, lambda3, P, zs, inv_method, save_a1a2=True)
            self.vm = get_Omega_GS2SLS(
                w.sparse, lambda3, self, moments_i[0], vc3, P)
            self.betas = np.vstack((tsls_s.betas, lambda3))
            self.e_filtered = self.u - lambda3 * lag_spatial(w, self.u)
            self.name_x = USER.set_name_x(name_x, x, constant=True)
            self.name_yend = USER.set_name_yend(name_yend, yend)
            self.name_z = self.name_x + self.name_yend
            self.name_z.append('lambda')  # listing lambda last
            self.name_q = USER.set_name_q(name_q, q)
            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}
            if summ:
                self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HET) - REGIMES"
                SUMMARY.GM_Endog_Error_Het(reg=self, w=w, vm=vm, regimes=True)

    def _endog_error_regimes_multi(self, y, x, regimes, w, yend, q, cores,
                                   max_iter, epsilon, step1c, inv_method, cols2regi, vm,
                                   name_x, name_yend, name_q, add_lag):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
        if add_lag != False:
            self.cols2regi += [True]
            cols2regi += [True]
            self.predy_e = np.zeros((self.n, 1), float)
            self.e_pred = np.zeros((self.n, 1), float)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work_endog_error(*(y,x,yend,q,regi_ids,r,w,max_iter,epsilon,step1c,inv_method,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag))
            else:
                pool = mp.Pool(cores)        
                results_p[r] = pool.apply_async(_work_endog_error,args=(y,x,yend,q,regi_ids,r,w,max_iter,epsilon,step1c,inv_method,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag, ))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work_endog_error, args=(y, x, yend, q, regi_ids, r, w, max_iter, epsilon, step1c,
                                                                         inv_method, self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes, add_lag, ))
            else:
                results_p[r] = _work_endog_error(*(y, x, yend, q, regi_ids, r, w, max_iter, epsilon, step1c, inv_method,
                                                   self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes, add_lag))

        self.kryd, self.kf = 0, 0
        self.kr = len(cols2regi) + 1
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
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
                (counter * self.kr):((counter + 1) * self.kr), ] = results[r].betas
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
            SUMMARY.GM_Combo_Het_multi(
                reg=self, multireg=self.multi, vm=vm, regimes=True)
        else:
            SUMMARY.GM_Endog_Error_Het_multi(
                reg=self, multireg=self.multi, vm=vm, regimes=True)


class GM_Combo_Het_Regimes(GM_Endog_Error_Het_Regimes):

    """
    GMM method for a spatial lag and error model with heteroskedasticity,
    regimes and endogenous variables, with results and diagnostics;
    based on Arraiz et al [1]_, following Anselin [2]_.

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
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from Arraiz et al. 
    inv_method   : string
                   If "power_exp", then compute inverse using the power
                   expansion. If "true_inv", then compute the true inverse.
                   Note that true_inv will fail for large n.
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
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
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
    multi         : dictionary
                    Only available when multiple regressions are estimated,
                    i.e. when regime_err_sep=True and no variable is fixed
                    across regimes.
                    Contains all attributes of each individual regression

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    .. [2] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity

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

    Since we want to run a spatial combo model, we need to specify
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

    Example only with spatial lag

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.  We can have a 
    summary of the output by typing: model.summary 
    Alternatively, we can check the betas:

    >>> reg = GM_Combo_Het_Regimes(y, x, regimes, w=w, step1c=True, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')
    >>> print reg.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '_Global_W_HR90', 'lambda']
    >>> print np.around(reg.betas,4)
    [[ 1.4613]
     [ 0.9587]
     [ 0.5658]
     [ 9.1157]
     [ 1.1324]
     [ 0.6518]
     [-0.4587]
     [ 0.7174]]

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

    >>> reg = GM_Combo_Het_Regimes(y, x, regimes, yd, q, w=w, step1c=True, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT')
    >>> print reg.name_z
    ['0_CONSTANT', '0_PS90', '0_UE90', '1_CONSTANT', '1_PS90', '1_UE90', '0_RD90', '1_RD90', '_Global_W_HR90', 'lambda']
    >>> print reg.betas
    [[ 3.41936197]
     [ 1.04071048]
     [ 0.16747219]
     [ 8.85820215]
     [ 1.847382  ]
     [-0.24545394]
     [ 2.43189808]
     [ 3.61328423]
     [ 0.03132164]
     [ 0.29544224]]
    >>> print np.sqrt(reg.vm.diagonal())
    [ 0.53103804  0.20835827  0.05755679  1.00496234  0.34332131  0.10259525
      0.3454436   0.37932794  0.07611667  0.07067059]
    >>> print 'lambda: ', np.around(reg.betas[-1], 4)
    lambda:  [ 0.2954]

    """

    def __init__(self, y, x, regimes, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True,
                 max_iter=1, epsilon=0.00001, step1c=False,
                 cores=False, inv_method='power_exp',
                 constant_regi='many', cols2regi='all',
                 regime_err_sep=False, regime_lag_sep=False,
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None, name_regimes=None):

        n = USER.check_arrays(y, x)
        self.step1c = step1c
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
            cols2regi += [False]
            add_lag = False
            if regime_err_sep == True:
                raise Exception, "For spatial combo models, if spatial error is set by regimes (regime_err_sep=True), all coefficients including lambda (regime_lag_sep=True) must be set by regimes."
            yend, q = set_endog(y, x, w, yend, q, w_lags, lag_q)
        name_yend.append(USER.set_name_yend_sp(self.name_y))

        GM_Endog_Error_Het_Regimes.__init__(self, y=y, x=x, yend=yend,
                                            q=q, regimes=regimes, w=w, constant_regi=constant_regi,
                                            cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                            max_iter=max_iter, epsilon=epsilon,
                                            step1c=step1c, inv_method=inv_method, cores=cores,
                                            vm=vm, name_y=name_y, name_x=name_x, name_yend=name_yend,
                                            name_q=name_q, name_w=name_w, name_ds=name_ds,
                                            name_regimes=name_regimes, summ=False, add_lag=add_lag)

        if regime_err_sep != True:
            self.rho = self.betas[-2]
            self.predy_e, self.e_pred, warn = UTILS.sp_att(w, self.y,
                                                           self.predy, yend[:, -1].reshape(self.n, 1), self.rho)
            UTILS.set_warn(self, warn)
            self.regime_lag_sep = regime_lag_sep
            self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HET) - REGIMES"
            SUMMARY.GM_Combo_Het(reg=self, w=w, vm=vm, regimes=True)


def _work_error(y, x, regi_ids, r, w, max_iter, epsilon, step1c, name_ds, name_y, name_x, name_w, name_regimes):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    model = BaseGM_Error_Het(
        y_r, x_constant, w_r.sparse, max_iter=max_iter, epsilon=epsilon, step1c=step1c)
    set_warn(model, warn)
    model.w = w_r
    model.title = "SPATIALLY WEIGHTED LEAST SQUARES ESTIMATION (HET) - REGIME %s" % r
    model.name_ds = name_ds
    model.name_y = '%s_%s' % (str(r), name_y)
    model.name_x = ['%s_%s' % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    return model


def _work_endog_error(y, x, yend, q, regi_ids, r, w, max_iter, epsilon, step1c, inv_method, name_ds, name_y, name_x, name_yend, name_q, name_w, name_regimes, add_lag):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    if yend != None:
        yend_r = yend[regi_ids[r]]
        q_r = q[regi_ids[r]]
    else:
        yend_r, q_r = None, None
    if add_lag != False:
        yend_r, q_r = set_endog(
            y_r, x_r, w_r, yend_r, q_r, add_lag[0], add_lag[1])
    x_constant = USER.check_constant(x_r)
    model = BaseGM_Endog_Error_Het(y_r, x_constant, yend_r, q_r, w_r.sparse,
                                   max_iter=max_iter, epsilon=epsilon, step1c=step1c, inv_method=inv_method)
    set_warn(model, warn)
    if add_lag != False:
        model.rho = model.betas[-2]
        model.predy_e, model.e_pred, warn = sp_att(w_r, model.y,
                                                   model.predy, model.yend[:, -1].reshape(model.n, 1), model.rho)
        set_warn(model, warn)
    model.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HET) - REGIME %s" % r
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
