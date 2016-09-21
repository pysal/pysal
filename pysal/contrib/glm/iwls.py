import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from pysal.spreg.utils import spdot, spmultiply
from family import Binomial

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
    #spx = sp.csr_matrix(x)
    #dx = np.float(spx.nnz)/np.float(np.multiply(*spx.shape))
    n_iter = 0
    diff = 1.0e6
    if isinstance(family, Binomial):
        y = family.link._clean(y)
    
    if ini_betas is None:
        betas = np.zeros((x.shape[1], 1), np.float)
    else:
        betas = ini_betas
    mu = family.starting_mu(y)
    v = family.predict(mu)
    while diff > tol and n_iter < max_iter:
        n_iter += 1
        w = family.weights(mu)
        z = v + (family.link.deriv(mu)*(y-mu))
        w = np.sqrt(w)
        if type(x) != np.ndarray:
        	w = sp.csr_matrix(w)
        	z = sp.csr_matrix(z)
        wx = spmultiply(x, w, array_out=False)
        wz = spmultiply(z, w, array_out=False)
        if wi is None:
            n_betas = _compute_betas(wz, wx)
        else:
            n_betas, xtx_inv_xt = _compute_betas_gwr(wz, wx, wi)
        v = spdot(x, n_betas)
        mu  = family.fitted(v)
        diff = min(abs(n_betas-betas))
        betas = n_betas
        
    if wi is None:
        return betas, mu, wx, n_iter
    else:
        return betas, mu, v, w, z, xtx_inv_xt, n_iter
