'''
Hom family of models based on: [Drukker2013]_ 
Following: [Anselin2011]_

'''

__author__ = "Luc Anselin luc.anselin@asu.edu, Daniel Arribas-Bel darribas@asu.edu"

from scipy import sparse as SP
import numpy as np
from numpy import linalg as la
import ols as OLS
from pysal import lag_spatial
from utils import power_expansion, set_endog, iter_msg, sp_att
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments
from utils import get_spFilter, get_lags, _moments2eqs
from utils import spdot, RegressionPropsY, set_warn
import twosls as TSLS
import user_output as USER
import summary_output as SUMMARY

__all__ = ["GM_Error_Hom", "GM_Endog_Error_Hom", "GM_Combo_Hom"]


class BaseGM_Error_Hom(RegressionPropsY):

    '''
    GMM method for a spatial error model with homoskedasticity (note: no
    consistency checks, diagnostics or constant added); based on 
    Drukker et al. (2013) [Drukker2013]_, following Anselin (2011) [Anselin2011]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix   
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011) (default).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).

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
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    xtx          : float
                   X'X

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
    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'

    Model commands

    >>> reg = BaseGM_Error_Hom(y, X, w=w.sparse, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9479  12.3021]
     [  0.7063   0.4967]
     [ -0.556    0.179 ]
     [  0.4129   0.1835]]
    >>> print np.around(reg.vm, 4) #doctest: +SKIP
    [[  1.51340700e+02  -5.29060000e+00  -1.85650000e+00  -2.40000000e-03]
     [ -5.29060000e+00   2.46700000e-01   5.14000000e-02   3.00000000e-04]
     [ -1.85650000e+00   5.14000000e-02   3.21000000e-02  -1.00000000e-04]
     [ -2.40000000e-03   3.00000000e-04  -1.00000000e-04   3.37000000e-02]]
    '''

    def __init__(self, y, x, w,
                 max_iter=1, epsilon=0.00001, A1='hom_sc'):
        if A1 == 'hom':
            wA1 = get_A1_hom(w)
        elif A1 == 'hom_sc':
            wA1 = get_A1_hom(w, scalarKP=True)
        elif A1 == 'het':
            wA1 = get_A1_het(w)

        wA2 = get_A2_hom(w)

        # 1a. OLS --> \tilde{\delta}
        ols = OLS.BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, wA1, wA2, ols.u)
        lambda1 = optim_moments(moments)
        lambda_old = lambda1

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. SWLS --> \hat{\delta}
            x_s = get_spFilter(w, lambda_old, self.x)
            y_s = get_spFilter(w, lambda_old, self.y)
            ols_s = OLS.BaseOLS(y=y_s, x=x_s)
            self.predy = spdot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, wA1, wA2, self.u)
            psi = get_vc_hom(w, wA1, wA2, self, lambda_old)[0]
            lambda2 = optim_moments(moments, psi)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            self.iteration += 1

        self.iter_stop = iter_msg(self.iteration, max_iter)

        # Output
        self.betas = np.vstack((ols_s.betas, lambda2))
        self.vm, self.sig2 = get_omega_hom_ols(
            w, wA1, wA2, self, lambda2, moments[0])
        self.e_filtered = self.u - lambda2 * w * self.u
        self._cache = {}


class GM_Error_Hom(BaseGM_Error_Hom):

    '''
    GMM method for a spatial error model with homoskedasticity, with results
    and diagnostics; based on Drukker et al. (2013) [Drukker2013]_, following Anselin
    (2011) [Anselin2011]_.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object   
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).
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
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    xtx          : float
                   X'X
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

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and CRIME (crime) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Error_Hom(y, X, w=w, A1='hom_sc', name_y='home value', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that assumes
    homoskedasticity but that unlike the models from
    ``pysal.spreg.error_sp``, it allows for inference on the spatial
    parameter. This is why you obtain as many coefficient estimates as
    standard errors, which you calculate taking the square root of the
    diagonal of the variance-covariance matrix of the parameters:

    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9479  12.3021]
     [  0.7063   0.4967]
     [ -0.556    0.179 ]
     [  0.4129   0.1835]]

    '''

    def __init__(self, y, x, w,
                 max_iter=1, epsilon=0.00001, A1='hom_sc',
                 vm=False, name_y=None, name_x=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        x_constant = USER.check_constant(x)
        BaseGM_Error_Hom.__init__(self, y=y, x=x_constant, w=w.sparse, A1=A1,
                                  max_iter=max_iter, epsilon=epsilon)
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES (HOM)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Error_Hom(reg=self, w=w, vm=vm)


class BaseGM_Endog_Error_Hom(RegressionPropsY):

    '''
    GMM method for a spatial error model with homoskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on Drukker et al. (2013) [Drukker2013]_, following Anselin (2011)
    [Anselin2011]_.

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
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).

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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    hth          : float
                   H'H

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> reg = BaseGM_Endog_Error_Hom(y, X, yd, q, w=w.sparse, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]


    '''

    def __init__(self, y, x, yend, q, w,
                 max_iter=1, epsilon=0.00001, A1='hom_sc'):

        if A1 == 'hom':
            wA1 = get_A1_hom(w)
        elif A1 == 'hom_sc':
            wA1 = get_A1_hom(w, scalarKP=True)
        elif A1 == 'het':
            wA1 = get_A1_het(w)

        wA2 = get_A2_hom(w)

        # 1a. S2SLS --> \tilde{\delta}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)
        self.x, self.z, self.h, self.y, self.hth = tsls.x, tsls.z, tsls.h, tsls.y, tsls.hth
        self.yend, self.q, self.n, self.k = tsls.yend, tsls.q, tsls.n, tsls.k

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, wA1, wA2, tsls.u)
        lambda1 = optim_moments(moments)
        lambda_old = lambda1

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. GS2SLS --> \hat{\delta}
            x_s = get_spFilter(w, lambda_old, self.x)
            y_s = get_spFilter(w, lambda_old, self.y)
            yend_s = get_spFilter(w, lambda_old, self.yend)
            tsls_s = TSLS.BaseTSLS(y=y_s, x=x_s, yend=yend_s, h=self.h)
            self.predy = spdot(self.z, tsls_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, wA1, wA2, self.u)
            psi = get_vc_hom(w, wA1, wA2, self, lambda_old, tsls_s.z)[0]
            lambda2 = optim_moments(moments, psi)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            self.iteration += 1

        self.iter_stop = iter_msg(self.iteration, max_iter)

        # Output
        self.betas = np.vstack((tsls_s.betas, lambda2))
        self.vm, self.sig2 = get_omega_hom(
            w, wA1, wA2, self, lambda2, moments[0])
        self.e_filtered = self.u - lambda2 * w * self.u
        self._cache = {}


class GM_Endog_Error_Hom(BaseGM_Endog_Error_Hom):

    '''
    GMM method for a spatial error model with homoskedasticity and endogenous
    variables, with results and diagnostics; based on Drukker et al. (2013)
    [Drukker2013]_, following Anselin (2011) [Anselin2011]_.

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
                   Spatial weights object   
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).
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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    hth          : float
                   H'H


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

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case we consider CRIME (crime rates) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for CRIME. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = pysal.rook_from_shapefile(pysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Endog_Error_Hom(y, X, yd, q, w=w, A1='hom_sc', name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that assumes
    homoskedasticity but that unlike the models from
    ``pysal.spreg.error_sp``, it allows for inference on the spatial
    parameter. Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]

    '''

    def __init__(self, y, x, yend, q, w,
                 max_iter=1, epsilon=0.00001, A1='hom_sc',
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        x_constant = USER.check_constant(x)
        BaseGM_Endog_Error_Hom.__init__(
            self, y=y, x=x_constant, w=w.sparse, yend=yend, q=q,
            A1=A1, max_iter=max_iter, epsilon=epsilon)
        self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HOM)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Endog_Error_Hom(reg=self, w=w, vm=vm)


class BaseGM_Combo_Hom(BaseGM_Endog_Error_Hom):

    '''
    GMM method for a spatial lag and error model with homoskedasticity and
    endogenous variables (note: no consistency checks, diagnostics or constant
    added); based on Drukker et al. (2013) [Drukker2013]_, following Anselin (2011)
    [Anselin2011]_.

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
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from Arraiz
                   et al. Note: epsilon provides an additional stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. Note: max_iter provides
                   an additional stop condition.
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).


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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    hth          : float
                   H'H


    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
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

    >>> reg = BaseGM_Combo_Hom(y, X, yend=yd2, q=q2, w=w.sparse, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2871]
     [  1.5683   0.4407]
     [  0.1513   0.4048]
     [  0.2103   0.4226]]


    Example with both spatial lag and other endogenous variables

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> yd2, q2 = pysal.spreg.utils.set_endog(y, X, w, yd, q, w_lags, True)
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = BaseGM_Combo_Hom(y, X, yd2, q2, w=w.sparse, A1='hom_sc')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.7705' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''

    def __init__(self, y, x, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True,
                 max_iter=1, epsilon=0.00001, A1='hom_sc'):

        BaseGM_Endog_Error_Hom.__init__(
            self, y=y, x=x, w=w, yend=yend, q=q, A1=A1,
            max_iter=max_iter, epsilon=epsilon)


class GM_Combo_Hom(BaseGM_Combo_Hom):

    '''
    GMM method for a spatial lag and error model with homoskedasticity and
    endogenous variables, with results and diagnostics; based on Drukker et
    al. (2013) [Drukker2013]_, following Anselin (2011) [Anselin2011]_.

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
                   Spatial weights object (always necessary)   
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
    A1           : string
                   If A1='het', then the matrix A1 is defined as in Arraiz et
                   al. If A1='hom', then as in Anselin (2011).  If
                   A1='hom_sc' (default), then as in Drukker, Egger and Prucha (2010)
                   and Drukker, Prucha and Raciborski (2010).
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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al.
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
    hth          : float
                   H'H


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

    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
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
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    Example only with spatial lag

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Combo_Hom(y, X, w=w, A1='hom_sc', name_x=['inc'],\
            name_y='hoval', name_yend=['crime'], name_q=['discbd'],\
            name_ds='columbus')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2871]
     [  1.5683   0.4407]
     [  0.1513   0.4048]
     [  0.2103   0.4226]]

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include CRIME (crime rates) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:


    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    And then we can run and explore the model analogously to the previous combo:

    >>> reg = GM_Combo_Hom(y, X, yd, q, w=w, A1='hom_sc', \
            name_ds='columbus')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.7705' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''

    def __init__(self, y, x, yend=None, q=None,
                 w=None, w_lags=1, lag_q=True,
                 max_iter=1, epsilon=0.00001, A1='hom_sc',
                 vm=False, name_y=None, name_x=None,
                 name_yend=None, name_q=None,
                 name_w=None, name_ds=None):

        n = USER.check_arrays(y, x, yend, q)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        x_constant = USER.check_constant(x)
        BaseGM_Combo_Hom.__init__(
            self, y=y, x=x_constant, w=w.sparse, yend=yend2, q=q2,
            w_lags=w_lags, A1=A1, lag_q=lag_q,
            max_iter=max_iter, epsilon=epsilon)
        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = sp_att(w, self.y, self.predy,
                                                 yend2[:, -1].reshape(self.n, 1), self.rho)
        set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES (HOM)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  # listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(
            USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.GM_Combo_Hom(reg=self, w=w, vm=vm)


# Functions

def moments_hom(w, wA1, wA2, u):
    '''
    Compute G and g matrices for the spatial error model with homoscedasticity
    as in Anselin [Anselin2011]_ (2011).
    ...

    Parameters
    ----------

    w           : Sparse matrix
                  Spatial weights sparse matrix   

    u           : array
                  Residuals. nx1 array assumed to be aligned with w

    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.


    '''
    n = w.shape[0]
    A1u = wA1 * u
    A2u = wA2 * u
    wu = w * u

    g1 = np.dot(u.T, A1u)
    g2 = np.dot(u.T, A2u)
    g = np.array([[g1][0][0], [g2][0][0]]) / n

    G11 = 2 * np.dot(wu.T * wA1, u)
    G12 = -np.dot(wu.T * wA1, wu)
    G21 = 2 * np.dot(wu.T * wA2, u)
    G22 = -np.dot(wu.T * wA2, wu)
    G = np.array([[G11[0][0], G12[0][0]], [G21[0][0], G22[0][0]]]) / n
    return [G, g]


def get_vc_hom(w, wA1, wA2, reg, lambdapar, z_s=None, for_omegaOLS=False):
    '''
    VC matrix \psi of Spatial error with homoscedasticity. As in 
    Anselin (2011) [Anselin2011]_ (p. 20)
    ...

    Parameters
    ----------
    w               :   Sparse matrix
                        Spatial weights sparse matrix
    reg             :   reg
                        Regression object
    lambdapar       :   float
                        Spatial parameter estimated in previous step of the
                        procedure
    z_s             :   array
                        optional argument for spatially filtered Z (to be
                        passed only if endogenous variables are present)
    for_omegaOLS    :   boolean
                        If True (default=False), it also returns P, needed
                        only in the computation of Omega

    Returns
    -------

    psi         : array
                  2x2 VC matrix
    a1          : array
                  nx1 vector a1. If z_s=None, a1 = 0.
    a2          : array
                  nx1 vector a2. If z_s=None, a2 = 0.
    p           : array
                  P matrix. If z_s=None or for_omegaOLS=False, p=0.

    '''
    u_s = get_spFilter(w, lambdapar, reg.u)
    n = float(w.shape[0])
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s ** 3) / n
    mu4 = np.sum(u_s ** 4) / n

    tr11 = wA1 * wA1
    tr11 = np.sum(tr11.diagonal())
    tr12 = wA1 * (wA2 * 2)
    tr12 = np.sum(tr12.diagonal())
    tr22 = wA2 * wA2 * 2
    tr22 = np.sum(tr22.diagonal())
    vecd1 = np.array([wA1.diagonal()]).T

    psi11 = 2 * sig2 ** 2 * tr11 + \
        (mu4 - 3 * sig2 ** 2) * np.dot(vecd1.T, vecd1)
    psi12 = sig2 ** 2 * tr12
    psi22 = sig2 ** 2 * tr22

    a1, a2, p = 0., 0., 0.

    if for_omegaOLS:
        x_s = get_spFilter(w, lambdapar, reg.x)
        p = la.inv(spdot(x_s.T, x_s) / n)

    if issubclass(type(z_s), np.ndarray) or \
            issubclass(type(z_s), SP.csr.csr_matrix) or \
            issubclass(type(z_s), SP.csc.csc_matrix):
        alpha1 = (-2 / n) * spdot(z_s.T, wA1 * u_s)
        alpha2 = (-2 / n) * spdot(z_s.T, wA2 * u_s)

        hth = spdot(reg.h.T, reg.h)
        hthni = la.inv(hth / n)
        htzsn = spdot(reg.h.T, z_s) / n
        p = spdot(hthni, htzsn)
        p = spdot(p, la.inv(spdot(htzsn.T, p)))
        hp = spdot(reg.h, p)
        a1 = spdot(hp, alpha1)
        a2 = spdot(hp, alpha2)

        psi11 = psi11 + \
            sig2 * spdot(a1.T, a1) + \
            2 * mu3 * spdot(a1.T, vecd1)
        psi12 = psi12 + \
            sig2 * spdot(a1.T, a2) + \
            mu3 * spdot(a2.T, vecd1)  # 3rd term=0
        psi22 = psi22 + \
            sig2 * spdot(a2.T, a2)  # 3rd&4th terms=0 bc vecd2=0

    psi = np.array(
        [[psi11[0][0], psi12[0][0]], [psi12[0][0], psi22[0][0]]]) / n
    return psi, a1, a2, p


def get_omega_hom(w, wA1, wA2, reg, lamb, G):
    '''
    Omega VC matrix for Hom models with endogenous variables computed as in
    Anselin (2011) [Anselin2011]_ (p. 21).
    ...

    Parameters
    ----------
    w       :   Sparse matrix
                Spatial weights sparse matrix
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    '''
    n = float(w.shape[0])
    z_s = get_spFilter(w, lamb, reg.z)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s ** 3) / n
    vecdA1 = np.array([wA1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, wA1, wA2, reg, lamb, z_s)
    j = np.dot(G, np.array([[1.], [2 * lamb]]))
    psii = la.inv(psi)
    t2 = spdot(reg.h.T, np.hstack((a1, a2)))
    psiDL = (mu3 * spdot(reg.h.T, np.hstack((vecdA1, np.zeros((n, 1))))) +
             sig2 * spdot(reg.h.T, np.hstack((a1, a2)))) / n

    oDD = spdot(la.inv(spdot(reg.h.T, reg.h)), spdot(reg.h.T, z_s))
    oDD = sig2 * la.inv(spdot(z_s.T, spdot(reg.h, oDD)))
    oLL = la.inv(spdot(j.T, spdot(psii, j))) / n
    oDL = spdot(spdot(spdot(p.T, psiDL), spdot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower)), float(sig2)


def get_omega_hom_ols(w, wA1, wA2, reg, lamb, G):
    '''
    Omega VC matrix for Hom models without endogenous variables (OLS) computed
    as in Anselin (2011) [Anselin2011]_.
    ...

    Parameters
    ----------
    w       :   Sparse matrix
                Spatial weights sparse matrix
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    '''
    n = float(w.shape[0])
    x_s = get_spFilter(w, lamb, reg.x)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    vecdA1 = np.array([wA1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, wA1, wA2, reg, lamb, for_omegaOLS=True)
    j = np.dot(G, np.array([[1.], [2 * lamb]]))
    psii = la.inv(psi)

    oDD = sig2 * la.inv(spdot(x_s.T, x_s))
    oLL = la.inv(spdot(j.T, spdot(psii, j))) / n
    #oDL = np.zeros((oDD.shape[0], oLL.shape[1]))
    mu3 = np.sum(u_s ** 3) / n
    psiDL = (mu3 * spdot(reg.x.T, np.hstack((vecdA1, np.zeros((n, 1)))))) / n
    oDL = spdot(spdot(spdot(p.T, psiDL), spdot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower)), float(sig2)


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':

    _test()
