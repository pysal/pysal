import numpy as np
from scipy import linalg
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from pysal.model.spreg.utils import spdot, spmultiply
from .family import Binomial, Poisson


def _compute_betas(y, x):
    """
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
    xT = x.T
    xtx = spdot(xT, x)
    xtx_inv = la.inv(xtx)
    xtx_inv = sp.csr_matrix(xtx_inv)
    xTy = spdot(xT, y, array_out=False)
    betas = spdot(xtx_inv, xTy)
    return betas


def _compute_betas_gwr(y, x, wi):
    """
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
    xT = (x * wi).T
    xtx = np.dot(xT, x)
    xtx_inv_xt = linalg.solve(xtx, xT)
    betas = np.dot(xtx_inv_xt, y)
    return betas, xtx_inv_xt


def iwls(y, x, family, offset, y_fix,
         ini_betas=None, tol=1.0e-8, max_iter=200, wi=None):
    """
    Iteratively re-weighted least squares estimation routine

    Parameters
    ----------
    y           : array
                  n*1, dependent variable

    x           : array
                  n*k, designs matrix of k independent variables

    family      : family object
                  probability models: Gaussian, Poisson, or Binomial

    offset      : array
                  n*1, the offset variable for each observation.

    y_fix       : array
                  n*1, the fixed intercept value of y for each observation

    ini_betas   : array
                  1*k, starting values for the k betas within the iteratively
                  weighted least squares routine

    tol         : float
                  tolerance for estimation convergence

    max_iter    : integer maximum number of iterations if convergence not met

    wi          : array
                  n*1, weights to transform observations from location i in GWR



    Returns
    -------

    betas       : array
                  k*1, estimated coefficients

    mu          : array
                  n*1, predicted y values

    wx          : array
                  n*1, final weights used for iwls for GLM

    n_iter      : integer
                  number of iterations that when iwls algorithm terminates

    w           : array
                  n*1, final weights used for iwls for GWR

    z           : array
                  iwls throughput

    v           : array
                  iwls throughput

    xtx_inv_xt  : array
                  iwls throughout to compute GWR hat matrix
                  [X'X]^-1 X'

    """
    n_iter = 0
    diff = 1.0e6

    if ini_betas is None:
        betas = np.zeros((x.shape[1], 1), np.float)
    else:
        betas = ini_betas

    if isinstance(family, Binomial):
        y = family.link._clean(y)
    if isinstance(family, Poisson):
        y_off = y / offset
        y_off = family.starting_mu(y_off)
        v = family.predict(y_off)
        mu = family.starting_mu(y)
    else:
        mu = family.starting_mu(y)
        v = family.predict(mu)

    while diff > tol and n_iter < max_iter:
        n_iter += 1
        w = family.weights(mu)
        z = v + (family.link.deriv(mu) * (y - mu))
        w = np.sqrt(w)
        if not isinstance(x, np.ndarray):
            w = sp.csr_matrix(w)
            z = sp.csr_matrix(z)
        wx = spmultiply(x, w, array_out=False)
        wz = spmultiply(z, w, array_out=False)
        if wi is None:
            n_betas = _compute_betas(wz, wx)
        else:
            n_betas, xtx_inv_xt = _compute_betas_gwr(wz, wx, wi)
        v = spdot(x, n_betas)
        mu = family.fitted(v)

        if isinstance(family, Poisson):
            mu = mu * offset

        diff = min(abs(n_betas - betas))
        betas = n_betas

    if wi is None:
        return betas, mu, wx, n_iter
    else:
        return betas, mu, v, w, z, xtx_inv_xt, n_iter
