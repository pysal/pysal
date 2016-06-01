"""Ordinary Least Squares regression classes."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu"
import numpy as np
import copy as COPY
import numpy.linalg as la
import user_output as USER
import summary_output as SUMMARY
import robust as ROBUST
from utils import spdot, sphstack, RegressionPropsY, RegressionPropsVM

__all__ = ["OLS"]


class BaseOLS(RegressionPropsY, RegressionPropsVM):

    """
    Ordinary least squares (OLS) (note: no consistency checks, diagnostics or
    constant added)

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
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
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    xtx          : float
                   X'X
    xtxi         : float
                   (X'X)^-1


    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> ols=BaseOLS(y,X)
    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> ols.vm
    array([[ 174.02245348,   -6.52060364,   -2.15109867],
           [  -6.52060364,    0.28720001,    0.06809568],
           [  -2.15109867,    0.06809568,    0.03336939]])
    """

    def __init__(self, y, x, robust=None, gwk=None, sig2n_k=True):
        self.x = x
        self.xtx = spdot(self.x.T, self.x)
        xty = spdot(self.x.T, y)

        self.xtxi = la.inv(self.xtx)
        self.betas = np.dot(self.xtxi, xty)
        predy = spdot(self.x, self.betas)

        u = y - predy
        self.u = u
        self.predy = predy
        self.y = y
        self.n, self.k = self.x.shape

        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

        if robust is not None:
            self.vm = ROBUST.robust_vm(reg=self, gwk=gwk, sig2n_k=sig2n_k)


class OLS(BaseOLS):

    """
    Ordinary least squares with results and diagnostics.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
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
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    robust       : string
                   Adjustment for robust standard errors
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    r2           : float
                   R squared
    ar2          : float
                   Adjusted R squared
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2ML       : float
                   Sigma squared (maximum likelihood)
    f_stat       : tuple
                   Statistic (float), p-value (float)
    logll        : float
                   Log likelihood
    aic          : float
                   Akaike information criterion 
    schwarz      : float
                   Schwarz information criterion     
    std_err      : array
                   1xk array of standard errors of the betas    
    t_stat       : list of tuples
                   t statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    mulColli     : float
                   Multicollinearity condition number
    jarque_bera  : dictionary
                   'jb': Jarque-Bera statistic (float); 'pvalue': p-value
                   (float); 'df': degrees of freedom (int)  
    breusch_pagan : dictionary
                    'bp': Breusch-Pagan statistic (float); 'pvalue': p-value
                    (float); 'df': degrees of freedom (int)  
    koenker_bassett : dictionary
                      'kb': Koenker-Bassett statistic (float); 'pvalue':
                      p-value (float); 'df': degrees of freedom (int)  
    white         : dictionary
                    'wh': White statistic (float); 'pvalue': p-value (float);
                    'df': degrees of freedom (int)  
    lm_error      : tuple
                    Lagrange multiplier test for spatial error model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
    lm_lag        : tuple
                    Lagrange multiplier test for spatial lag model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float 
    rlm_error     : tuple
                    Robust lagrange multiplier test for spatial error model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
    rlm_lag       : tuple
                    Robust lagrange multiplier test for spatial lag model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float
    lm_sarma      : tuple
                    Lagrange multiplier test for spatial SARMA model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float
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
    title         : string
                    Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    xtx          : float
                   X'X
    xtxi         : float
                   (X'X)^-1


    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; also, the actual OLS class
    requires data to be passed in as numpy arrays so the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an nx1 numpy array.

    >>> hoval = db.by_col("HOVAL")
    >>> y = np.array(hoval)
    >>> y.shape = (len(hoval), 1)

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). pysal.spreg.OLS adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    The minimum parameters needed to run an ordinary least squares regression
    are the two numpy arrays containing the independent variable and dependent
    variables respectively.  To make the printed results more meaningful, the
    user can pass in explicit names for the variables used; this is optional.

    >>> ols = OLS(y, X, name_y='home value', name_x=['income','crime'], name_ds='columbus', white_test=True)

    pysal.spreg.OLS computes the regression coefficients and their standard
    errors, t-stats and p-values. It also computes a large battery of
    diagnostics on the regression. In this example we compute the white test
    which by default isn't ('white_test=True'). All of these results can be independently
    accessed as attributes of the regression object created by running
    pysal.spreg.OLS.  They can also be accessed at one time by printing the
    summary attribute of the regression object. In the example below, the
    parameter on crime is -0.4849, with a t-statistic of -2.6544 and p-value
    of 0.01087.

    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> print round(ols.t_stat[2][0],3)
    -2.654
    >>> print round(ols.t_stat[2][1],3)
    0.011
    >>> print round(ols.r2,3)
    0.35

    Or we can easily obtain a full summary of all the results nicely formatted and
    ready to be printed:

    >>> print ols.summary
    REGRESSION
    ----------
    SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES
    -----------------------------------------
    Data set            :    columbus
    Dependent Variable  :  home value                Number of Observations:          49
    Mean dependent var  :     38.4362                Number of Variables   :           3
    S.D. dependent var  :     18.4661                Degrees of Freedom    :          46
    R-squared           :      0.3495
    Adjusted R-squared  :      0.3212
    Sum squared residual:   10647.015                F-statistic           :     12.3582
    Sigma-square        :     231.457                Prob(F-statistic)     :   5.064e-05
    S.E. of regression  :      15.214                Log likelihood        :    -201.368
    Sigma-square ML     :     217.286                Akaike info criterion :     408.735
    S.E of regression ML:     14.7406                Schwarz criterion     :     414.411
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     t-Statistic     Probability
    ------------------------------------------------------------------------------------
                CONSTANT      46.4281827      13.1917570       3.5194844       0.0009867
                   crime      -0.4848885       0.1826729      -2.6544086       0.0108745
                  income       0.6289840       0.5359104       1.1736736       0.2465669
    ------------------------------------------------------------------------------------
    <BLANKLINE>
    REGRESSION DIAGNOSTICS
    MULTICOLLINEARITY CONDITION NUMBER           12.538
    <BLANKLINE>
    TEST ON NORMALITY OF ERRORS
    TEST                             DF        VALUE           PROB
    Jarque-Bera                       2          39.706           0.0000
    <BLANKLINE>
    DIAGNOSTICS FOR HETEROSKEDASTICITY
    RANDOM COEFFICIENTS
    TEST                             DF        VALUE           PROB
    Breusch-Pagan test                2           5.767           0.0559
    Koenker-Bassett test              2           2.270           0.3214
    <BLANKLINE>
    SPECIFICATION ROBUST TEST
    TEST                             DF        VALUE           PROB
    White                             5           2.906           0.7145
    ================================ END OF REPORT =====================================

    If the optional parameters w and spat_diag are passed to pysal.spreg.OLS,
    spatial diagnostics will also be computed for the regression.  These
    include Lagrange multiplier tests and Moran's I of the residuals.  The w
    parameter is a PySAL spatial weights matrix. In this example, w is built
    directly from the shapefile columbus.shp, but w can also be read in from a
    GAL or GWT file.  In this case a rook contiguity weights matrix is built,
    but PySAL also offers queen contiguity, distance weights and k nearest
    neighbor weights among others. In the example, the Moran's I of the
    residuals is 0.204 with a standardized value of 2.592 and a p-value of
    0.0095.

    >>> w = pysal.weights.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> ols = OLS(y, X, w, spat_diag=True, moran=True, name_y='home value', name_x=['income','crime'], name_ds='columbus')
    >>> ols.betas
    array([[ 46.42818268],
           [  0.62898397],
           [ -0.48488854]])
    >>> print round(ols.moran_res[0],3)
    0.204
    >>> print round(ols.moran_res[1],3)
    2.592
    >>> print round(ols.moran_res[2],4)
    0.0095

    """

    def __init__(self, y, x,
                 w=None,
                 robust=None, gwk=None, sig2n_k=True,
                 nonspat_diag=True, spat_diag=False, moran=False,
                 white_test=False, vm=False, name_y=None, name_x=None,
                 name_w=None, name_gwk=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        x_constant = USER.check_constant(x)
        BaseOLS.__init__(self, y=y, x=x_constant, robust=robust,
                         gwk=gwk, sig2n_k=sig2n_k)
        self.title = "ORDINARY LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        SUMMARY.OLS(reg=self, vm=vm, w=w, nonspat_diag=nonspat_diag,
                    spat_diag=spat_diag, moran=moran, white_test=white_test)


def _test():
    import doctest
    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    #doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
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
    x_var = ['INC', 'HOVAL']
    x = np.array([db.by_col(name) for name in x_var]).T
    w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    ols = OLS(
        y, x, w=w, nonspat_diag=True, spat_diag=True, name_y=y_var, name_x=x_var,
        name_ds='columbus', name_w='columbus.gal', robust='white', sig2n_k=True, moran=True)
    print ols.summary
