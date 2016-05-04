import numpy as np
import numpy.linalg as la

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

def iwls(x, y, g_ey, link_func, family, offset, y_fix,
    ini_betas=None, tol=1.0e-6, max_iter=200, wi=None):
    """
    Iteratively re-weighted least squares estimation routine

    """
    diff = 1.0e6
    n_iter = 0
    if ini_betas is None:
        betas = _compute_betas(g_ey-y_fix, x)
    else:
        betas = ini_betas

    v = np.dot(x, betas)
    while diff > tol and n_iter < max_iter:
        n_iter += 1
        z, w = link_func(v, y, offset, y_fix)
        w = np.sqrt(w)
        wx = x * w
        wz = z * w
        if wi is None:
            n_betas = _compute_betas(wz, wx)
        else:
            n_betas, xtx_inv_xt = _compute_betas_gwr(wz, wx, wi)
        v_new = np.dot(x, n_betas)

	if family == 'Gaussian':
	    diff = 0.0
	else:
	    diff = min(abs(n_betas-betas))

        v = v_new
        betas = n_betas

    n_iter += 1

    if wi is None:
        return betas, w, v, n_iter
    else:
        return betas, w, v, n_iter, z, xtx_inv_xt
