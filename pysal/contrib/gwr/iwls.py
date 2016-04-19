import numpy as np
import numpy.linalg as la

def _compute_betas(y, x):
    """
    compute MLE coefficients using iwls routine
        
    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS), 
    Fotheringham, Brunsdon and Charlton (2002)
    """ 
    xtx = np.dot(x.T, x)
    xtx_inv = la.inv(xtx)
    xtxi = xtx_inv
    xtx_inv_xt = np.dot(xtx_inv, x.T)
    betas = np.dot(xtx_inv_xt, y)
    return betas   

def iwls(x, y, g_ey, link_func, family, offset, y_fix, sMatrix=None,
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
                
        ww = np.sqrt(w)
        wx = x * ww
        if sMatrix is None:
            wz = z * ww
            n_betas = _compute_betas(wz, wx)
	    #else:
	        #wz = (z - np.dot(sMatrix,z)) * ww
	        #wx2 = (x - np.dot(sMatrix,x)) * ww
	        #xtx = np.dot(wx.T, wx2)
	        #xtxinv = la.inv(xtx)
	        #xtz = np.dot(wx.T, wz)
	        #n_betas = np.dot(xtxinv, xtz)
                
        v_new = np.dot(x, n_betas)
	if family == 'Gaussian':
	    diff = 0.0
	else:
	    diff = min(abs(n_betas-betas))
                
        v = v_new
        betas = n_betas

    n_iter += 1

    return betas, w, v, n_iter
