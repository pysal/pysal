"""
Spatial Error Models module
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
from numpy import linalg as la
import ols as OLS
from pysal import lag_spatial
from utils import power_expansion, set_endog, iter_msg, sp_att
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
from utils import spdot, RegressionPropsY, set_warn
import twosls as TSLS
import user_output as USER
import summary_output as SUMMARY

__all__ = ["GM_Error", "GM_Endog_Error", "GM_Combo"]


class BaseGM_Error(RegressionPropsY):

    """
    GMM method for a spatial error model (note: no consistency checks
    diagnostics or constant added); based on Kelejian and Prucha 
    (1998, 1999) [Kelejian1998]_ [Kelejian1999]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix   

    Attributes
    ----------
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
    sig2         : float
                   Sigma squared used in computations

    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('HOVAL')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 
    >>> w.transform='r'
    >>> model = BaseGM_Error(y, x, w=w.sparse)
    >>> np.around(model.betas, decimals=4)
    array([[ 47.6946],
           [  0.7105],
           [ -0.5505],
           [  0.3257]])
    """

    def __init__(self, y, x, w):

        # 1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y=y, x=x)
        self.n, self.k = ols.x.shape
        self.x = ols.x
        self.y = ols.y

        # 1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, ols.u)
        lambda1 = optim_moments(moments)

        # 2a. OLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        ols2 = OLS.BaseOLS(y=ys, x=xs)

        # Output
        self.predy = spdot(self.x, ols2.betas)
        self.u = y - self.predy
        self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
        self.sig2 = ols2.sig2n
        self.e_filtered = self.u - lambda1 * w * self.u

        self.vm = self.sig2 * ols2.xtxi
        se_betas = np.sqrt(self.vm.diagonal())
        self._cache = {}


class GM_Error(BaseGM_Error):

    """
    GMM method for a spatial error model, with results and diagnostics; based
    on Kelejian and Prucha (1998, 1999) [Kelejian1998]_ [Kelejian1999]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object (always needed)   
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
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
    e_filtered   : array
                   nx1 array of spatially filtered residuals
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
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('HOVAL')]).T

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> names_to_extract = ['INC', 'CRIME']
    >>> x = np.array([dbf.by_col(name) for name in names_to_extract]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error(y, x, w=w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas).

    >>> print model.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> np.around(model.betas, decimals=4)
    array([[ 47.6946],
           [  0.7105],
           [ -0.5505],
           [  0.3257]])
    >>> np.around(model.std_err, decimals=4)
    array([ 12.412 ,   0.5044,   0.1785])
    >>> np.around(model.z_stat, decimals=6) #doctest: +SKIP
    array([[  3.84261100e+00,   1.22000000e-04],
           [  1.40839200e+00,   1.59015000e-01],
           [ -3.08424700e+00,   2.04100000e-03]])
    >>> round(model.sig2,4)
    198.5596

    """

    def __init__(self, y, x, w,
                 vm=False, name_y=None, name_x=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        x_constant = USER.check_constant(x)
        BaseGM_Error.__init__(self, y=y, x=x_constant, w=w.sparse)
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Error(reg=self, w=w, vm=vm)


class BaseGM_Endog_Error(RegressionPropsY):

    '''
    GMM method for a spatial error model with endogenous variables (note: no
    consistency checks, diagnostics or constant added); based on Kelejian and
    Prucha (1998, 1999) [Kelejian1998]_ [Kelejian1999]_.

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
    w            : Sparse matrix
                   Spatial weights sparse matrix 

    Attributes
    ----------
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
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations

    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> yend = np.array([dbf.by_col('HOVAL')]).T
    >>> q = np.array([dbf.by_col('DISCBD')]).T
    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 
    >>> w.transform='r'
    >>> model = BaseGM_Endog_Error(y, x, yend, q, w=w.sparse)
    >>> np.around(model.betas, decimals=4)
    array([[ 82.573 ],
           [  0.581 ],
           [ -1.4481],
           [  0.3499]])

    '''

    def __init__(self, y, x, yend, q, w):

        # 1a. TSLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)
        self.n, self.k = tsls.z.shape
        self.x = tsls.x
        self.y = tsls.y
        self.yend, self.z = tsls.yend, tsls.z

        # 1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, tsls.u)
        lambda1 = optim_moments(moments)

        # 2a. 2SLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        yend_s = get_spFilter(w, lambda1, self.yend)
        tsls2 = TSLS.BaseTSLS(ys, xs, yend_s, h=tsls.h)

        # Output
        self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
        self.predy = spdot(tsls.z, tsls2.betas)
        self.u = y - self.predy
        self.sig2 = float(np.dot(tsls2.u.T, tsls2.u)) / self.n
        self.e_filtered = self.u - lambda1 * w * self.u
        self.vm = self.sig2 * tsls2.varb
        self._cache = {}


class GM_Endog_Error(BaseGM_Endog_Error):

    '''
    GMM method for a spatial error model with endogenous variables, with
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
    w            : pysal W object
                   Spatial weights object (always needed)   
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
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
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
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
    title         : string
                    Name of the regression method used

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> dbf = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('CRIME')]).T

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in.

    >>> x = np.array([dbf.by_col('INC')]).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yend = np.array([dbf.by_col('HOVAL')]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q = np.array([dbf.by_col('DISCBD')]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Endog_Error(y, x, yend, q, w=w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included.

    >>> print model.name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> np.around(model.betas, decimals=4)
    array([[ 82.573 ],
           [  0.581 ],
           [ -1.4481],
           [  0.3499]])
    >>> np.around(model.std_err, decimals=4)
    array([ 16.1381,   1.3545,   0.7862])

    '''

    def __init__(self, y, x, yend, q, w,
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        x_constant = USER.check_constant(x)
        BaseGM_Endog_Error.__init__(
            self, y=y, x=x_constant, w=w.sparse, yend=yend, q=q)
        self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Endog_Error(reg=self, w=w, vm=vm)


class BaseGM_Combo(BaseGM_Endog_Error):

    """
    GMM method for a spatial lag and error model, with endogenous variables
    (note: no consistency checks, diagnostics or constant added); based on 
    Kelejian and Prucha (1998, 1999) [Kelejian1998]_ [Kelejian1999]_.

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
    w            : Sparse matrix
                   Spatial weights sparse matrix  
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).

    Attributes
    ----------
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
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> w_lags = 1
    >>> yd2, q2 = pysal.spreg.utils.set_endog(y, X, w, None, None, w_lags, True)
    >>> X = np.hstack((np.ones(y.shape),X))

    Example only with spatial lag

    >>> reg = BaseGM_Combo(y, X, yend=yd2, q=q2, w=w.sparse)

    Print the betas

    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.059  11.86 ]
     [ -1.404   0.391]
     [  0.467   0.2  ]]

    And lambda

    >>> print 'Lamda: ', np.around(reg.betas[-1], 3)
    Lamda:  [-0.048]

    Example with both spatial lag and other endogenous variables

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> yd2, q2 = pysal.spreg.utils.set_endog(y, X, w, yd, q, w_lags, True)
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = BaseGM_Combo(y, X, yd2, q2, w=w.sparse)
    >>> betas = np.array([['CONSTANT'],['INC'],['HOVAL'],['W_CRIME']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['CONSTANT' '50.0944' '14.3593']
     ['INC' '-0.2552' '0.5667']
     ['HOVAL' '-0.6885' '0.3029']
     ['W_CRIME' '0.4375' '0.2314']]

        """

    def __init__(self, y, x, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True):

        BaseGM_Endog_Error.__init__(self, y=y, x=x, w=w, yend=yend, q=q)


class GM_Combo(BaseGM_Combo):

    """
    GMM method for a spatial lag and error model with endogenous variables,
    with results and diagnostics; based on Kelejian and Prucha (1998,
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
    w            : pysal W object
                   Spatial weights object (always needed)   
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
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    z            : array
                   nxk array of variables (combination of x and yend)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    sig2         : float
                   Sigma squared used in computations (based on filtered
                   residuals)
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
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
    title         : string
                    Name of the regression method used

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

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

    >>> reg = GM_Combo(y, X, w=w, name_y='crime', name_x=['income'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can check the betas:

    >>> print reg.name_z
    ['CONSTANT', 'income', 'W_crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.059  11.86 ]
     [ -1.404   0.391]
     [  0.467   0.2  ]]

    And lambda:

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [-0.048]

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include HOVAL (home value) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    And then we can run and explore the model analogously to the previous combo:

    >>> reg = GM_Combo(y, X, yd, q, w=w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> names = np.array(reg.name_z).reshape(5,1)
    >>> print np.hstack((names[0:4,:], np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['CONSTANT' '50.0944' '14.3593']
     ['inc' '-0.2552' '0.5667']
     ['hoval' '-0.6885' '0.3029']
     ['W_crime' '0.4375' '0.2314']]

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [ 0.254]

    """

    def __init__(self, y, x, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True,
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        x_constant = USER.check_constant(x)
        BaseGM_Combo.__init__(
            self, y=y, x=x_constant, w=w.sparse, yend=yend2, q=q2,
            w_lags=w_lags, lag_q=lag_q)
        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = sp_att(w, self.y,
                                                 self.predy, yend2[:, -1].reshape(self.n, 1), self.rho)
        set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(
            USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Combo(reg=self, w=w, vm=vm)


def _momentsGM_Error(w, u):
    try:
        wsparse = w.sparse
    except:
        wsparse = w
    n = wsparse.shape[0]
    u2 = np.dot(u.T, u)
    wu = wsparse * u
    uwu = np.dot(u.T, wu)
    wu2 = np.dot(wu.T, wu)
    wwu = wsparse * wu
    uwwu = np.dot(u.T, wwu)
    wwu2 = np.dot(wwu.T, wwu)
    wuwwu = np.dot(wu.T, wwu)
    wtw = wsparse.T * wsparse
    trWtW = np.sum(wtw.diagonal())
    g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / n
    G = np.array(
        [[2 * uwu[0][0], -wu2[0][0], n], [2 * wuwwu[0][0], -wwu2[0][0], trWtW],
         [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.]]) / n
    return [G, g]


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
    y = np.array([dbf.by_col('HOVAL')]).T
    names_to_extract = ['INC', 'CRIME']
    x = np.array([dbf.by_col(name) for name in names_to_extract]).T
    w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    w.transform = 'r'
    model = GM_Error(y, x, w, name_y='hoval',
                     name_x=['income', 'crime'], name_ds='columbus')
    print model.summary
