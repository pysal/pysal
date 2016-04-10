"""
Diagnostics for regression estimations. 
"""
__author__ = "Luc Anselin luc.anselin@asu.edu, Nicholas Malizia nicholas.malizia@asu.edu"

import numpy as np

__all__ = [
    "f_stat", "t_stat", "r2", "ar2", "se_betas", "log_likelihood", "akaike", "schwarz",
    "condition_index", "jarque_bera", "breusch_pagan", "white", "koenker_bassett", "vif", "likratiotest"]


def constant_check(array):
    """
    Checks to see numpy array includes a constant.

    Parameters
    ----------
    array           : array
                      an array of variables to be inspected 

    Returns
    -------
    constant        : boolean
                      true signifies the presence of a constant

    Example
    -------

    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics
    >>> from ols import OLS
    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> diagnostics.constant_check(reg.x)
    True

    """

    n, k = array.shape
    constant = False
    for j in range(k):
        variable = array[:, j]
        varmin = variable.min()
        varmax = variable.max()
        if varmin == varmax:
            constant = True
            break
    return constant


def likratiotest(reg0, reg1):
    """
