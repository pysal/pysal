import numpy as np
import pysal
import scipy.sparse as SP
import itertools as iter
from scipy.stats import f, chisqprob
import numpy.linalg as la
from utils import spbroadcast

"""
Tools for different regimes procedure estimations
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu"


class Chow:

    '''
    Chow test of coefficient stability across regimes. The test is a
    particular case of the Wald statistic in which the constraint are setup
    according to the spatial or other type of regime structure

    ...

    Parameters
    ==========
    reg     : regression object
              Regression object from PySAL.spreg which is assumed to have the
              following attributes:

                    * betas : coefficient estimates
                    * vm    : variance covariance matrix of betas
                    * kr    : Number of variables varying across regimes
                    * kryd  : Number of endogenous variables varying across regimes
                    * kf    : Number of variables fixed (global) across regimes
                    * nr    : Number of regimes

    Attributes
    ==========
    joint   : tuple
              Pair of Wald statistic and p-value for the setup of global
              regime stability, that is all betas are the same across
              regimes.
    regi    : array
              kr x 2 array with Wald statistic (col 0) and its p-value (col 1)
              for each beta that varies across regimes. The restrictions
              are setup to test for the global stability (all regimes have the
              same parameter) of the beta.

    Examples
    ========
    >>> import numpy as np
    >>> import pysal
    >>> from ols_regimes import OLS_Regimes
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y_var = 'CRIME'
    >>> y = np.array([db.by_col(y_var)]).reshape(49,1)
    >>> x_var = ['INC','HOVAL']
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> r_var = 'NSA'
    >>> regimes = db.by_col(r_var)
    >>> olsr = OLS_Regimes(y, x, regimes, constant_regi='many', nonspat_diag=False, spat_diag=False, name_y=y_var, name_x=x_var, name_ds='columbus', name_regimes=r_var, regime_err_sep=False)
    >>> print olsr.name_x_r #x_var
    ['CONSTANT', 'INC', 'HOVAL']
    >>> print olsr.chow.regi
    [[ 0.01020844  0.91952121]
     [ 0.46024939  0.49750745]
     [ 0.55477371  0.45637369]]
    >>> print 'Joint test:'
    Joint test:
    >>> print olsr.chow.joint
    (0.6339319928978806, 0.8886223520178802)
    '''

    def __init__(self, reg):
        kr, kf, kryd, nr, betas, vm = reg.kr, reg.kf, reg.kryd, reg.nr, reg.betas, reg.vm
        if betas.shape[0] != vm.shape[0]:
            if kf > 0:
                betas = betas[0:vm.shape[0], :]
                kf = kf - 1
            else:
                brange = []
                for i in range(nr):
                    brange.extend(range(i * (kr + 1), i * (kr + 1) + kr))
                betas = betas[brange, :]
        r_global = []
        regi = np.zeros((reg.kr, 2))
        for vari in np.arange(kr):
            r_vari = buildR1var(vari, kr, kf, kryd, nr)
            r_global.append(r_vari)
            q = np.zeros((r_vari.shape[0], 1))
            regi[vari, :] = wald_test(betas, r_vari, q, vm)
        r_global = np.vstack(tuple(r_global))
        q = np.zeros((r_global.shape[0], 1))
        joint = wald_test(betas, r_global, q, vm)
        self.joint = joint
        self.regi = regi


class Wald:

    '''
    Chi sq. Wald statistic to test for restriction of coefficients.
    Implementation following Greene [Greene2003]_ eq. (17-24), p. 488

    ...

    Parameters
    ==========
    reg     : regression object
              Regression object from PySAL.spreg
    r       : array
              Array of dimension Rxk (R being number of restrictions) with constrain setup.
    q       : array
              Rx1 array with constants in the constraint setup. See Greene
              [1]_ for reference.

    Attributes
    ==========
    w       : float
              Wald statistic
    pvalue  : float
              P value for Wald statistic calculated as a Chi sq. distribution
              with R degrees of freedom

    '''

    def __init__(self, reg, r, q=None):
        if not q:
            q = np.zeros((r.shape[0], 1))
        self.w, self.pvalue = wald_test(reg.betas, r, q, reg.vm)


class Regimes_Frame:

    '''
    Setup framework to work with regimes. Basically it involves:
        * Dealing with the constant in a regimes world
        * Creating a sparse representation of X 
        * Generating a list of names of X taking into account regimes

    ...

    Parameters
    ==========
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: [False, 'one', 'many']
                   Switcher controlling the constant term setup. It may take
                   the following values:

                     *  False: no constant term is appended in any way
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
    names         : None, list of strings
                   Names of independent variables for use in output

    Returns
    =======
    x            : csr sparse matrix
                   Sparse matrix containing X variables properly aligned for
                   regimes regression. 'xsp' is of dimension (n, k*r) where 'r'
                   is the number of different regimes
                   The structure of the alignent is X1r1 X2r1 ... X1r2 X2r2 ...
    names        : None, list of strings
                   Names of independent variables for use in output
                   conveniently arranged by regimes. The structure of the name
                   is "regimeName_-_varName"
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

    '''

    def __init__(self, x, regimes, constant_regi, cols2regi, names=None, yend=False):
        if cols2regi == 'all':
            cols2regi = [True] * x.shape[1]
        else:
            if yend:
                cols2regi = cols2regi[-x.shape[1]:]
            else:
                cols2regi = cols2regi[0:x.shape[1]]
        if constant_regi:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            if constant_regi == 'one':
                cols2regi.insert(0, False)
            elif constant_regi == 'many':
                cols2regi.insert(0, True)
            else:
                raise Exception, "Invalid argument (%s) passed for 'constant_regi'. Please secify a valid term." % str(
                    constant)
        try:
            x = regimeX_setup(x, regimes, cols2regi,
                              self.regimes_set, constant=constant_regi)
        except AttributeError:
            self.regimes_set = _get_regimes_set(regimes)
            x = regimeX_setup(x, regimes, cols2regi,
                              self.regimes_set, constant=constant_regi)

        kr = len(np.where(np.array(cols2regi) == True)[0])
        if yend:
            self.kr += kr
            self.kf += len(cols2regi) - kr
            self.kryd = kr
        else:
            self.kr = kr
            self.kf = len(cols2regi) - self.kr
            self.kryd = 0
        self.nr = len(set(regimes))

        if names:
            names = set_name_x_regimes(
                names, regimes, constant_regi, cols2regi, self.regimes_set)

        return (x, names)


def wald_test(betas, r, q, vm):
    '''
    Chi sq. Wald statistic to test for restriction of coefficients.
    Implementation following Greene [Greene2003]_ eq. (17-24), p. 488

    ...

    Parameters
    ==========
    betas   : array
              kx1 array with coefficient estimates
    r       : array
              Array of dimension Rxk (R being number of restrictions) with constrain setup.
    q       : array
              Rx1 array with constants in the constraint setup. See Greene
              [1]_ for reference.
    vm      : array
              kxk variance-covariance matrix of coefficient estimates

    Returns
    =======
    w       : float
              Wald statistic
    pvalue  : float
              P value for Wald statistic calculated as a Chi sq. distribution
              with R degrees of freedom

    '''
    rbq = np.dot(r, betas) - q
    rvri = la.inv(np.dot(r, np.dot(vm, r.T)))
    w = np.dot(rbq.T, np.dot(rvri, rbq))[0][0]
    df = r.shape[0]
    pvalue = chisqprob(w, df)
    return w, pvalue


def buildR(kr, kf, nr):
    '''
    Build R matrix to globally test for spatial heterogeneity across regimes.
    The constraint setup reflects the null every beta is the same
    across regimes
    
    Note: needs a placeholder for kryd in builR1var, set to 0

    ...

    Parameters
    ==========
    kr      : int
              Number of variables that vary across regimes ("regimized")
    kf      : int
              Number of variables that do not vary across regimes ("fixed" or
              global)
    nr      : int
              Number of regimes

    Returns
    =======
    R       : array
              Array with constrain setup to test stability across regimes of
              one variable
    '''
    return np.vstack(tuple(map(buildR1var, np.arange(kr), [kr] * kr, [kf] * kr,\
               [0] * kr, [nr] * kr)))


def buildR1var(vari, kr, kf, kryd, nr):
    '''
    Build R matrix to test for spatial heterogeneity across regimes in one
    variable. The constraint setup reflects the null betas for variable 'vari'
    are the same across regimes

    ...

    Parameters
    ==========
    vari    : int
              Position of the variable to be tested (order in the sequence of
              variables per regime)
    kr      : int
              Number of variables that vary across regimes ("regimized")
    kf      : int
              Number of variables that do not vary across regimes ("fixed" or
              global)
    kryd    : Number of endogenous variables varying across regimes
    nr      : int
              Number of regimes

    Returns
    =======
    R       : array
              Array with constrain setup to test stability across regimes of
              one variable
    '''
    ncols = (kr * nr)
    nrows = nr - 1
    r = np.zeros((nrows, ncols), dtype=int)
    rbeg = 0
    krexog = kr - kryd
    if vari < krexog:
        kr_j = krexog
        cbeg = vari
    else:
        kr_j = kryd
        cbeg = krexog * (nr - 1) + vari
    r[rbeg: rbeg + nrows, cbeg] = 1
    for j in np.arange(nrows):
        r[rbeg + j, kr_j + cbeg + j * kr_j] = -1
    return np.hstack((r, np.zeros((nrows, kf), dtype=int)))


def regimeX_setup(x, regimes, cols2regi, regimes_set, constant=False):
    '''
    Flexible full setup of a regime structure

    NOTE: constant term, if desired in the model, should be included in the x
    already

    ...

    Parameters
    ==========
    x           : np.array
                  Dense array of dimension (n, k) with values for all observations
                  IMPORTANT: constant term (if desired in the model) should be
                  included
    regimes     : list
                  list of n values with the mapping of each observation to a
                  regime. Assumed to be aligned with 'x'.
    cols2regi   : list
                  List of k booleans indicating whether each column should be
                  considered as different per regime (True) or held constant
                  across regimes (False)
    regimes_set : list
                  List of ordered regimes tags
    constant    : [False, 'one', 'many']
                  Switcher controlling the constant term setup. It may take
                  the following values:

                    *  False: no constant term is appended in any way
                    *  'one': a vector of ones is appended to x and held
                              constant across regimes
                    * 'many': a vector of ones is appended to x and considered
                              different per regime

    Returns
    =======
    xsp         : csr sparse matrix
                  Sparse matrix containing the full setup for a regimes model
                  as specified in the arguments passed
                  NOTE: columns are reordered so first are all the regime
                  columns then all the global columns (this makes it much more
                  efficient)
                  Structure of the output matrix (assuming X1, X2 to vary
                  across regimes and constant term, X3 and X4 to be global):
                    X1r1, X2r1, ... , X1r2, X2r2, ... , constant, X3, X4
    '''
    cols2regi = np.array(cols2regi)
    if set(cols2regi) == set([True]):
        xsp = x2xsp(x, regimes, regimes_set)
    elif set(cols2regi) == set([False]):
        xsp = SP.csr_matrix(x)
    else:
        not_regi = x[:, np.where(cols2regi == False)[0]]
        regi_subset = x[:, np.where(cols2regi)[0]]
        regi_subset = x2xsp(regi_subset, regimes, regimes_set)
        xsp = SP.hstack((regi_subset, SP.csr_matrix(not_regi)), format='csr')
    return xsp


def set_name_x_regimes(name_x, regimes, constant_regi, cols2regi, regimes_set):
    '''
    Generate the set of variable names in a regimes setup, according to the
    order of the betas

    NOTE: constant term, if desired in the model, should be included in the x
    already

    ...

    Parameters
    ==========
    name_x          : list/None
                      If passed, list of strings with the names of the
                      variables aligned with the original dense array x
                      IMPORTANT: constant term (if desired in the model) should be
                      included
    regimes         : list
                      list of n values with the mapping of each observation to a
                      regime. Assumed to be aligned with 'x'.
    constant_regi   : [False, 'one', 'many']
                      Switcher controlling the constant term setup. It may take
                      the following values:

                         *  False: no constant term is appended in any way
                         *  'one': a vector of ones is appended to x and held
                                   constant across regimes
                         * 'many': a vector of ones is appended to x and considered
                                   different per regime
    cols2regi       : list
                      List of k booleans indicating whether each column should be
                      considered as different per regime (True) or held constant
                      across regimes (False)
    regimes_set     : list
                      List of ordered regimes tags
    Returns
    =======
    name_x_regi
    '''
    k = len(cols2regi)
    if constant_regi:
        k -= 1
    if not name_x:
        name_x = ['var_' + str(i + 1) for i in range(k)]
    if constant_regi:
        name_x.insert(0, 'CONSTANT')
    nxa = np.array(name_x)
    c2ra = np.array(cols2regi)
    vars_regi = nxa[np.where(c2ra == True)]
    vars_glob = nxa[np.where(c2ra == False)]
    name_x_regi = []
    for r in regimes_set:
        rl = ['%s_%s' % (str(r), i) for i in vars_regi]
        name_x_regi.extend(rl)
    name_x_regi.extend(['_Global_%s' % i for i in vars_glob])
    return name_x_regi


def w_regime(w, regi_ids, regi_i, transform=True, min_n=None):
    '''
    Returns the subset of W matrix according to a given regime ID

    ...

    Attributes
    ==========
    w           : pysal W object
                  Spatial weights object
    regi_ids    : list
                  Contains the location of observations in y that are assigned to regime regi_i
    regi_i      : string or float
                  The regime for which W will be subset

    Returns
    =======
    w_regi_i    : pysal W object
                  Subset of W for regime regi_i
    '''
    w_ids = map(w.id_order.__getitem__, regi_ids)
    warn = None
    w_regi_i = pysal.weights.w_subset(w, w_ids, silent_island_warning=True)
    if min_n:
        if w_regi_i.n < min_n:
            raise Exception, "There are less observations than variables in regime %s." % regi_i
    if transform:
        w_regi_i.transform = w.get_transform()
    if w_regi_i.islands:
        warn = "The regimes operation resulted in islands for regime %s." % regi_i
    return w_regi_i, warn


def w_regimes(w, regimes, regimes_set, transform=True, get_ids=None, min_n=None):
    '''
    ######### DEPRECATED ##########
    Subsets W matrix according to regimes

    ...

    Attributes
    ==========
    w           : pysal W object
                  Spatial weights object
    regimes     : list
                  list of n values with the mapping of each observation to a
                  regime. Assumed to be aligned with 'x'.
    regimes_set : list
                  List of ordered regimes tags

    Returns
    =======
    w_regi      : dictionary
                  Dictionary containing the subsets of W according to regimes: [r1:w1, r2:w2, ..., rR:wR]
    '''
    regi_ids = dict((r, list(np.where(np.array(regimes) == r)[0]))
                    for r in regimes_set)
    w_ids = dict((r, map(w.id_order.__getitem__, regi_ids[r]))
                 for r in regimes_set)
    w_regi_i = {}
    warn = None
    for r in regimes_set:
        w_regi_i[r] = pysal.weights.w_subset(w, w_ids[r],
                                             silent_island_warning=True)
        if min_n:
            if w_regi_i[r].n < min_n:
                raise Exception, "There are less observations than variables in regime %s." % r
        if transform:
            w_regi_i[r].transform = w.get_transform()
        if w_regi_i[r].islands:
            warn = "The regimes operation resulted in islands for regime %s." % r
    if get_ids:
        get_ids = regi_ids
    return w_regi_i, get_ids, warn


def w_regimes_union(w, w_regi_i, regimes_set):
    '''
    Combines the subsets of the W matrix according to regimes

    ...

    Attributes
    ==========
    w           : pysal W object
                  Spatial weights object
    w_regi_i    : dictionary
                  Dictionary containing the subsets of W according to regimes: [r1:w1, r2:w2, ..., rR:wR]
    regimes_set : list
                  List of ordered regimes tags

    Returns
    =======
    w_regi      : pysal W object
                  Spatial weights object containing the union of the subsets of W
    '''
    w_regi = pysal.weights.w_union(w_regi_i[regimes_set[0]],
                                   w_regi_i[regimes_set[1]], silent_island_warning=True)
    if len(regimes_set) > 2:
        for i in range(len(regimes_set))[2:]:
            w_regi = pysal.weights.w_union(w_regi,
                                           w_regi_i[regimes_set[i]], silent_island_warning=True)
    w_regi = pysal.weights.remap_ids(w_regi, dict((i, i)
                                                  for i in w_regi.id_order), w.id_order)
    w_regi.transform = w.get_transform()
    return w_regi


def x2xsp(x, regimes, regimes_set):
    '''
    Convert X matrix with regimes into a sparse X matrix that accounts for the
    regimes

    ...

    Attributes
    ==========
    x           : np.array
                  Dense array of dimension (n, k) with values for all observations
    regimes     : list
                  list of n values with the mapping of each observation to a
                  regime. Assumed to be aligned with 'x'.
    regimes_set : list
                  List of ordered regimes tags
    Returns
    =======
    xsp         : csr sparse matrix
                  Sparse matrix containing X variables properly aligned for
                  regimes regression. 'xsp' is of dimension (n, k*r) where 'r'
                  is the number of different regimes
                  The structure of the alignent is X1r1 X2r1 ... X1r2 X2r2 ...
    '''
    n, k = x.shape
    data = x.flatten()
    R = len(regimes_set)
    # X1r1 X2r1 ... X1r2 X2r2 ...
    regime_by_row = np.array([[r] * k for r in list(regimes_set)]).flatten()
    row_map = dict((r, np.where(regime_by_row == r)[0]) for r in regimes_set)
    indices = np.array([row_map[row] for row in regimes]).flatten()
    indptr = np.zeros((n + 1, ), dtype=int)
    indptr[:-1] = list(np.arange(n) * k)
    indptr[-1] = n * k
    return SP.csr_matrix((data, indices, indptr))


def check_cols2regi(constant_regi, cols2regi, x, yend=None, add_cons=True):
    ''' Checks if dimensions of list cols2regi match number of variables. '''

    if add_cons:
        is_cons = 1
        if constant_regi == 'many':
            regi_cons = [True]
        elif constant_regi == 'one':
            regi_cons = [False]
    else:
        is_cons = 0
        regi_cons = []
    try:
        tot_k = x.shape[1] + yend.shape[1]
    except:
        tot_k = x.shape[1]
    if cols2regi == 'all':
        cols2regi = regi_cons + [True] * tot_k
    else:
        cols2regi = regi_cons + cols2regi
    if len(cols2regi) - is_cons != tot_k:
        raise Exception, "The lenght of list 'cols2regi' must be equal to the amount of variables (exogenous + endogenous) when not using cols2regi=='all'."
    return cols2regi


def _get_regimes_set(regimes):
    ''' Creates a list with regimes in alphabetical order. '''
    regimes_set = list(set(regimes))
    if isinstance(regimes_set[0], float):
        regimes_set1 = list(set(map(int, regimes_set)))
        if len(regimes_set1) == len(regimes_set):
            regimes_set = regimes_set1
    regimes_set.sort()
    return regimes_set


def _get_weighted_var(regimes, regimes_set, sig2n_k, u, y, x, yend=None, q=None):
    regi_ids = dict((r, list(np.where(np.array(regimes) == r)[0]))
                    for r in regimes_set)
    if sig2n_k:
        sig = dict((r, np.dot(u[regi_ids[r]].T, u[regi_ids[r]]) / (len(regi_ids[r]) - x.shape[1]))
                   for r in regimes_set)
    else:
        sig = dict((r, np.dot(u[regi_ids[r]].T, u[regi_ids[r]]) / len(regi_ids[r]))
                   for r in regimes_set)
    sig_vec = np.zeros(y.shape, float)
    y2 = np.zeros(y.shape, float)
    for r in regimes_set:
        sig_vec[regi_ids[r]] = 1 / float(np.sqrt(sig[r]))
        y2[regi_ids[r]] = y[regi_ids[r]] / float(np.sqrt(sig[r]))
    x2 = spbroadcast(x, sig_vec)
    if yend != None:
        yend2 = spbroadcast(yend, sig_vec)
        q2 = spbroadcast(q, sig_vec)
        return y2, x2, yend2, q2
    else:
        return y2, x2


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
    from ols_regimes import OLS_Regimes
    db = pysal.open(pysal.examples.get_path('columbus.dbf'), 'r')
    y_var = 'CRIME'
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ['INC', 'HOVAL']
    x = np.array([db.by_col(name) for name in x_var]).T
    r_var = 'NSA'
    regimes = db.by_col(r_var)
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    olsr = OLS_Regimes(
        y, x, regimes, w=w, constant_regi='many', nonspat_diag=False, spat_diag=False,
        name_y=y_var, name_x=x_var, name_ds='columbus', name_regimes=r_var, name_w='columbus.gal')
    print olsr.summary
