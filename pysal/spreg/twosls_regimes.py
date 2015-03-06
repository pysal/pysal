import numpy as np
import regimes as REGI
import user_output as USER
import multiprocessing as mp
import scipy.sparse as SP
from utils import sphstack, set_warn, RegressionProps_basic, spdot, sphstack
from twosls import BaseTSLS
from robust import hac_multi
import summary_output as SUMMARY
from platform import system

"""
Two-stage Least Squares estimation with regimes.
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, David C. Folch david.folch@asu.edu"


class TSLS_Regimes(BaseTSLS, REGI.Regimes_Frame):

    """
    Two stage least squares (2SLS) with regimes

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
    vm           : boolean
                   If True, include variance-covariance matrix in summary
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
    name_regimes : string
                   Name of regimes variable for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
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
    vm           : array
                   Variance covariance matrix (kxk)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: [False, 'one', 'many']
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
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
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

    In this case we consider RD90 (resource deprivation) as an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for RD90. We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and 
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to perform tests for spatial dependence, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will
    create one from ``NAT.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("NAT.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We can now run the regression and then have a summary of the output
    by typing: model.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> tslsr = TSLS_Regimes(y, x, yd, q, regimes, w=w, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT', name_w='NAT.shp')

    >>> tslsr.betas
    array([[ 3.66973562],
           [ 1.06950466],
           [ 0.14680946],
           [ 2.45864196],
           [ 9.55873243],
           [ 1.94666348],
           [-0.30810214],
           [ 3.68718119]])

    >>> np.sqrt(tslsr.vm.diagonal())
    array([ 0.38389901,  0.09963973,  0.04672091,  0.22725012,  0.49181223,
            0.19630774,  0.07784587,  0.25529011])

    """

    def __init__(self, y, x, yend, q, regimes,
                 w=None, robust=None, gwk=None, sig2n_k=True,
                 spat_diag=False, vm=False, constant_regi='many',
                 cols2regi='all', regime_err_sep=True, name_y=None, name_x=None,
                 cores=False, name_yend=None, name_q=None, name_regimes=None,
                 name_w=None, name_gwk=None, name_ds=None, summ=True):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        self.name_x_r = USER.set_name_x(name_x, x) + name_yend
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x, yend=yend, add_cons=False)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x.shape[1])
        if regime_err_sep == True and robust == 'hac':
            set_warn(
                self, "Error by regimes is incompatible with HAC estimation for 2SLS models. Hence, the error by regimes has been disabled for this model.")
            regime_err_sep = False
        self.regime_err_sep = regime_err_sep
        if regime_err_sep == True and set(cols2regi) == set([True]) and constant_regi == 'many':
            name_x = USER.set_name_x(name_x, x)
            self.y = y
            regi_ids = dict(
                (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set)
            self._tsls_regimes_multi(x, yend, q, w, regi_ids, cores,
                                     gwk, sig2n_k, robust, spat_diag, vm, name_x, name_yend, name_q)
        else:
            name_x = USER.set_name_x(name_x, x, constant=True)
            q, self.name_q = REGI.Regimes_Frame.__init__(self, q,
                                                         regimes, constant_regi=None, cols2regi='all', names=name_q)
            x, self.name_x = REGI.Regimes_Frame.__init__(self, x,
                                                         regimes, constant_regi, cols2regi=cols2regi, names=name_x)
            yend, self.name_yend = REGI.Regimes_Frame.__init__(self, yend,
                                                               regimes, constant_regi=None,
                                                               cols2regi=cols2regi, yend=True, names=name_yend)
            if regime_err_sep == True and robust == None:
                robust = 'white'
            BaseTSLS.__init__(self, y=y, x=x, yend=yend, q=q,
                              robust=robust, gwk=gwk, sig2n_k=sig2n_k)
            self.title = "TWO STAGE LEAST SQUARES - REGIMES"
            if robust == 'ogmm':
                _optimal_weight(self, sig2n_k)
            self.name_z = self.name_x + self.name_yend
            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.chow = REGI.Chow(self)
            self.robust = USER.set_robust(robust)
            if summ:
                SUMMARY.TSLS(
                    reg=self, vm=vm, w=w, spat_diag=spat_diag, regimes=True)

    def _tsls_regimes_multi(self, x, yend, q, w, regi_ids, cores,
                            gwk, sig2n_k, robust, spat_diag, vm, name_x, name_yend, name_q):
        results_p = {}
        """
        for r in self.regimes_set:
            if system() != 'Windows':
                is_win = True
                results_p[r] = _work(*(self.y,x,w,regi_ids,r,yend,q,robust,sig2n_k,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work,args=(self.y,x,w,regi_ids,r,yend,q,robust,sig2n_k,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes))
                is_win = False
        """
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(_work, args=(
                    self.y, x, w, regi_ids, r, yend, q, robust, sig2n_k, self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes))
            else:
                results_p[r] = _work(*(self.y, x, w, regi_ids, r, yend, q, robust, sig2n_k,
                                       self.name_ds, self.name_y, name_x, name_yend, name_q, self.name_w, self.name_regimes))

        self.kryd = 0
        self.kr = x.shape[1] + yend.shape[1] + 1
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
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            counter += 1
        self.multi = results
        self.hac_var = sphstack(x, q)
        if robust == 'hac':
            hac_multi(self, gwk)
        if robust == 'ogmm':
            set_warn(
                self, "Residuals treated as homoskedastic for the purpose of diagnostics.")
        self.chow = REGI.Chow(self)
        if spat_diag:
            self._get_spat_diag_props(results, regi_ids, x, yend, q)
        SUMMARY.TSLS_multi(
            reg=self, multireg=self.multi, vm=vm, spat_diag=spat_diag, regimes=True, w=w)

    def _get_spat_diag_props(self, results, regi_ids, x, yend, q):
        self._cache = {}
        x = USER.check_constant(x)
        x = REGI.regimeX_setup(
            x, self.regimes, [True] * x.shape[1], self.regimes_set)
        self.z = sphstack(x, REGI.regimeX_setup(
            yend, self.regimes, [True] * yend.shape[1], self.regimes_set))
        self.h = sphstack(
            x, REGI.regimeX_setup(q, self.regimes, [True] * q.shape[1], self.regimes_set))
        hthi = np.linalg.inv(spdot(self.h.T, self.h))
        zth = spdot(self.z.T, self.h)
        self.varb = np.linalg.inv(spdot(spdot(zth, hthi), zth.T))


def _work(y, x, w, regi_ids, r, yend, q, robust, sig2n_k, name_ds, name_y, name_x, name_yend, name_q, name_w, name_regimes):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    yend_r = yend[regi_ids[r]]
    q_r = q[regi_ids[r]]
    x_constant = USER.check_constant(x_r)
    if robust == 'hac' or robust == 'ogmm':
        robust2 = None
    else:
        robust2 = robust
    model = BaseTSLS(
        y_r, x_constant, yend_r, q_r, robust=robust2, sig2n_k=sig2n_k)
    model.title = "TWO STAGE LEAST SQUARES ESTIMATION - REGIME %s" % r
    if robust == 'ogmm':
        _optimal_weight(model, sig2n_k, warn=False)
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
    if w:
        w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
        set_warn(model, warn)
        model.w = w_r
    return model


def _optimal_weight(reg, sig2n_k, warn=True):
    try:
        Hu = reg.h.toarray() * reg.u ** 2
    except:
        Hu = reg.h * reg.u ** 2
    if sig2n_k:
        S = spdot(reg.h.T, Hu, array_out=True) / (reg.n - reg.k)
    else:
        S = spdot(reg.h.T, Hu, array_out=True) / reg.n
    Si = np.linalg.inv(S)
    ZtH = spdot(reg.z.T, reg.h)
    ZtHSi = spdot(ZtH, Si)
    fac2 = np.linalg.inv(spdot(ZtHSi, ZtH.T, array_out=True))
    fac3 = spdot(ZtHSi, spdot(reg.h.T, reg.y), array_out=True)
    betas = np.dot(fac2, fac3)
    if sig2n_k:
        vm = fac2 * (reg.n - reg.k)
    else:
        vm = fac2 * reg.n
    RegressionProps_basic(reg, betas=betas, vm=vm, sig2=False)
    reg.title += " (Optimal-Weighted GMM)"
    if warn:
        set_warn(
            reg, "Residuals treated as homoskedastic for the purpose of diagnostics.")
    return


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
    db = pysal.open(pysal.examples.get_path('NAT.dbf'), 'r')
    y_var = 'HR60'
    y = np.array([db.by_col(y_var)]).T
    x_var = ['PS60', 'DV60', 'RD60']
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ['UE60']
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ['FP59', 'MA60']
    q = np.array([db.by_col(name) for name in q_var]).T
    r_var = 'SOUTH'
    regimes = db.by_col(r_var)
    tslsr = TSLS_Regimes(y, x, yd, q, regimes, constant_regi='many', spat_diag=False, name_y=y_var, name_x=x_var,
                         name_yend=yd_var, name_q=q_var, name_regimes=r_var, cols2regi=[
                             False, True, True, True],
                         sig2n_k=False)
    print tslsr.summary
