"""
Diagnostics for two stage least squares regression estimations. 
        
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Nicholas Malizia nicholas.malizia@asu.edu "

from pysal.common import *
from scipy.stats import pearsonr

__all__ = ["t_stat", "pr2_aspatial", "pr2_spatial"]


def t_stat(reg, z_stat=False):
    """
    Calculates the t-statistics (or z-statistics) and associated p-values.
    [Greene2003]_

    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
    z_stat          : boolean
                      If True run z-stat instead of t-stat

    Returns
    -------    
    ts_result       : list of tuples
                      each tuple includes value of t statistic (or z
                      statistic) and associated p-value

    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The ``diagnostics`` module is used for the tests
    we will show here and the OLS and TSLS are required to run the models on
    which we will perform the tests.

    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as diagnostics
    >>> from pysal.spreg.ols import OLS
    >>> from twosls import TSLS

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Before being able to apply the diagnostics, we have to run a model and,
    for that, we need the input variables. Extract the CRIME column (crime
    rates) from the DBF file and make it the dependent variable for the
    regression. Note that PySAL requires this to be an numpy array of shape
    (n, 1) as opposed to the also common shape of (n, ) that other packages
    accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and HOVAL (home value) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T

    Run an OLS regression. Since it is a non-spatial model, all we need is the
    dependent and the independent variable.

    >>> reg = OLS(y,X)

    Now we can perform a t-statistic on the model:

    >>> testresult = diagnostics.t_stat(reg)
    >>> print("%12.12f"%testresult[0][0], "%12.12f"%testresult[0][1], "%12.12f"%testresult[1][0], "%12.12f"%testresult[1][1], "%12.12f"%testresult[2][0], "%12.12f"%testresult[2][1])
    ('14.490373143689', '0.000000000000', '-4.780496191297', '0.000018289595', '-2.654408642718', '0.010874504910')

    We can also use the z-stat. For that, we re-build the model so we consider
    HOVAL as endogenous, instrument for it using DISCBD and carry out two
    stage least squares (TSLS) estimation.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T    
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Once the variables are read as different objects, we are good to run the
    model.

    >>> reg = TSLS(y, X, yd, q)

    With the output of the TSLS regression, we can perform a z-statistic:

    >>> testresult = diagnostics.t_stat(reg, z_stat=True)
    >>> print("%12.10f"%testresult[0][0], "%12.10f"%testresult[0][1], "%12.10f"%testresult[1][0], "%12.10f"%testresult[1][1], "%12.10f"%testresult[2][0], "%12.10f"%testresult[2][1])
    ('5.8452644705', '0.0000000051', '0.3676015668', '0.7131703463', '-1.9946891308', '0.0460767956')
    """

    k = reg.k           # (scalar) number of ind. vas (includes constant)
    n = reg.n           # (scalar) number of observations
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)
    betas = reg.betas   # (array) coefficients of the regressors (1 x k)
    variance = vm.diagonal()
    tStat = betas.reshape(len(betas),) / np.sqrt(variance)
    ts_result = []
    for t in tStat:
        if z_stat:
            ts_result.append((t, stats.norm.sf(abs(t)) * 2))
        else:
            ts_result.append((t, stats.t.sf(abs(t), n - k) * 2))
    return ts_result


def pr2_aspatial(tslsreg):
    """
    Calculates the pseudo r^2 for the two stage least squares regression.

    Parameters
    ----------
    tslsreg             : two stage least squares regression object
                          output instance from a two stage least squares
                          regression model

    Returns
    -------
    pr2_result          : float
                          value of the squared pearson correlation between
                          the y and tsls-predicted y vectors

    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The TSLS is required to run the model on
    which we will perform the tests.

    >>> import numpy as np
    >>> import pysal
    >>> from twosls import TSLS

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Before being able to apply the diagnostics, we have to run a model and,
    for that, we need the input variables. Extract the CRIME column (crime
    rates) from the DBF file and make it the dependent variable for the
    regression. Note that PySAL requires this to be an numpy array of shape
    (n, 1) as opposed to the also common shape of (n, ) that other packages
    accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case, we consider HOVAL (home value) as an endogenous regressor,
    so we acknowledge that by reading it in a different category.

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T

    In order to properly account for the endogeneity, we have to pass in the
    instruments. Let us consider DISCBD (distance to the CBD) is a good one:

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Now we are good to run the model. It is an easy one line task.

    >>> reg = TSLS(y, X, yd, q=q)

    In order to perform the pseudo R^2, we pass the regression object to the
    function and we are done!

    >>> result = pr2_aspatial(reg)
    >>> print("%1.6f"%result)    
    0.279361

    """

    y = tslsreg.y
    predy = tslsreg.predy
    pr = pearsonr(y, predy)[0]
    pr2_result = float(pr ** 2)
    return pr2_result


def pr2_spatial(tslsreg):
    """
    Calculates the pseudo r^2 for the spatial two stage least squares 
    regression.

    Parameters
    ----------
    stslsreg            : spatial two stage least squares regression object
                          output instance from a spatial two stage least 
                          squares regression model

    Returns
    -------    
    pr2_result          : float
                          value of the squared pearson correlation between
                          the y and stsls-predicted y vectors

    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The GM_Lag is required to run the model on
    which we will perform the tests and the ``pysal.spreg.diagnostics`` module
    contains the function with the test.

    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as D
    >>> from twosls_sp import GM_Lag

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')

    Extract the HOVAL column (home value) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))

    In this case, we consider CRIME (crime rates) as an endogenous regressor,
    so we acknowledge that by reading it in a different category.

    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))

    In order to properly account for the endogeneity, we have to pass in the
    instruments. Let us consider DISCBD (distance to the CBD) is a good one:

    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))

    Since this test has a spatial component, we need to specify the spatial
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

    Now we are good to run the spatial lag model. Make sure you pass all the
    parameters correctly and, if desired, pass the names of the variables as
    well so when you print the summary (reg.summary) they are included:

    >>> reg = GM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')

    Once we have a regression object, we can perform the spatial version of
    the pesudo R^2. It is as simple as one line!

    >>> result = pr2_spatial(reg)
    >>> print("%1.6f"%result)
    0.299649

    """

    y = tslsreg.y
    predy_e = tslsreg.predy_e
    pr = pearsonr(y, predy_e)[0]
    pr2_result = float(pr ** 2)
    return pr2_result


def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
