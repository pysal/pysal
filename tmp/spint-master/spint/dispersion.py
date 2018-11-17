"""
Various functions to test hypotheses regarding the dispersion of the variance of
a variable.

"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

from spglm.glm import GLM
from spglm.family import Poisson
import numpy as np
import scipy.stats as stats
from types import FunctionType


def phi_disp(model):
    """
    Test the hypothesis that var[y] = mu (equidispersion) against the
    alternative hypothesis (quasi-Poisson) that var[y] = phi * mu  where mu
    is the expected value of y and phi is an estimated overdispersion
    coefficient which is equivalent to 1+alpha in the alternative alpha
    dispersion test.

    phi > 0: overdispersion
    phi = 1: equidispersion
    phi < 0: underdispersion

    Parameters
    ----------
    model       : Model results class
                  function can only be called on a sucessfully fitted model
                  which has a valid response variable, y, and a valid
                  predicted response variable, yhat.
    alt_var     : function
                  specifies an alternative varaince as a function of mu.
                  Function must take a single scalar as input and return a
                  single scalar as output
    Returns
    -------
    array       : [alpha coefficient, tvalue of alpha, pvalue of alpha]

    """
    try:
        y = model.y.reshape((-1, 1))
        yhat = model.yhat.reshape((-1, 1))
        ytest = (((y - yhat)**2 - y) / yhat).reshape((-1, 1))
    except BaseException:
        raise AttributeError(
            "Check that fitted model has valid 'y' and 'yhat' attributes")

    phi = 1 + np.mean(ytest)
    zval = np.sqrt(len(ytest)) * np.mean(ytest) / np.std(ytest, ddof=1)
    pval = stats.norm.sf(zval)

    return np.array([phi, zval, pval])


def alpha_disp(model, alt_var=lambda x: x):
    """
    Test the hypothesis that var[y] = mu (equidispersion) against the
    alternative hypothesis that var[y] = mu + alpha * alt_var(mu) where mu
    is the expected value of y, alpha is an estimated coefficient, and
    alt_var() specifies an alternative variance as a function of mu.
    alt_var=lambda x:x corresponds to an alternative hypothesis of a negative
    binomimal model with a linear variance function and alt_var=lambda
    x:x**2 correspinds to an alternative hypothesis of a negative binomial
    model with a quadratic varaince function.

    alpha > 0: overdispersion
    alpha = 1: equidispersion
    alpha < 0: underdispersion

    Parameters
    ----------
    model       : Model results class
                  function can only be called on a sucessfully fitted model
                  which has a valid response variable, y, and a valid
                  predicted response variable, yhat.
    alt_var     : function
                  specifies an alternative varaince as a function of mu.
                  Function must take a single scalar as input and return a
                  single scalar as output
    Returns
    -------
    array       : [alpha coefficient, tvalue of alpha, pvalue of alpha]

    """
    try:
        y = model.y.reshape((-1, 1))
        yhat = model.yhat.reshape((-1, 1))
        ytest = (((y - yhat)**2 - y) / yhat).reshape((-1, 1))
    except BaseException:
        raise AttributeError(
            "Make sure model passed has been estimated and has a valid 'y' and 'yhat' attribute")

    if isinstance(alt_var, FunctionType):
        X = (alt_var(yhat) / yhat).reshape((-1, 1))
        test_results = GLM(ytest, X, constant=False).fit()
        alpha = test_results.params[0]
        zval = test_results.tvalues[0]
        pval = stats.norm.sf(zval)
    else:
        raise TypeError(
            "The alternative variance function, 'alt_var', must be a valid function'")

    return np.array([alpha, zval, pval])
