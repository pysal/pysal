import numpy as np
import numpy.linalg as la
from family import Binomial


def _compute_betas(y, x):
    """
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
    xT = x.T
    xtx = np.dot(xT, x)
    xtx_inv = la.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, xT)
    betas = np.dot(xtx_inv_xt, y)
    return betas

def _compute_betas_gwr(y, x, wi):
    """
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
    xT = np.dot(x.T, wi)
    xtx = np.dot(xT, x)
    xtx_inv = la.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, xT)
    betas = np.dot(xtx_inv_xt, y)
    return betas, xtx_inv_xt

def iwls(y, x, family, offset, y_fix,
    ini_betas=None, tol=1.0e-8, max_iter=200, wi=None):
    """
    Iteratively re-weighted least squares estimation routine

    """
    n_iter = 0
    diff = 1.0e6
    link_y = family.link(y)
    if isinstance(family, Binomial):
        y = family.link._clean(y)
    if ini_betas is None:
        #betas = _compute_betas(link_y, x)
        betas = np.zeros((x.shape[1], 1), np.float)
        mu = family.starting_mu(y)
        v = family.predict(mu)
    #else:
        #betas = ini_betas
    #v = np.dot(x, betas)
   
    while diff > tol and n_iter < max_iter:
    	n_iter += 1
        #mu = family.link.inverse(v)
        w = family.weights(mu)
        z = v + (family.link.deriv(mu)*(y-mu))
        w = np.sqrt(w)
        wx = x * w
        wz = z * w
        
        if wi is None:
            n_betas = _compute_betas(wz, wx)
        else:
            n_betas, xtx_inv_xt = _compute_betas_gwr(wz, wx, wi)
        
        v = np.dot(x, n_betas)
        mu  = family.fitted(v)

        diff = min(abs(n_betas-betas))
        betas = n_betas
        
    y_hat = mu
    if wi is None:
        return betas, y_hat, wx, n_iter
    else:
        return betas, y_hat, n_iter, xtx_inv_xt
