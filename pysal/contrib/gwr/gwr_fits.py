
import numpy as np
from glm_links import link_g, link_p, link_l, get_y_hat
from iwls import iwls

def gauss_iwls(GLM, wi):
    g_ey = GLM.y
    betas, w, v, n_iter, z, xtx_inv_xt = iwls(GLM.x, GLM.y, g_ey, link_g, GLM.family, GLM.offset,
            GLM.y_fix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['max_iter'], wi=wi)
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas.T, predy, v, w, z, xtx_inv_xt]

def poiss_iwls(GLM, wi):     
    ey = GLM.y/GLM.offset
    g_ey = np.log(ey)
    betas, w, v, n_iter, z, xtx_inv_xt = iwls(GLM.x, GLM.y+.001, g_ey, link_p, GLM.family, GLM.offset,
            GLM.y_fix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['max_iter'], wi=wi)
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas.T, predy, v, w, z, xtx_inv_xt]

def logit_iwls(GLM, wi):
    ey = GLM.y/GLM.offset
    theta = 1.0 * np.sum(GLM.y)/GLM.n
    id_one = GLM.y == 1
    id_zero = GLM.y == 0
    g_ey = np.ones(shape=(GLM.n,1))
    g_ey[id_one] = np.log(theta/(1.0 - theta))
    g_ey[id_zero] = np.log((1.0 - theta)/theta)
    betas, w, v, n_iter, z, xtx_inv_xt = iwls(GLM.x, GLM.y, g_ey, link_l, GLM.family, GLM.offset,
            GLM.y_fix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['max_iter'], wi=wi)
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas.T, predy, v, w, z, xtx_inv_xt]

