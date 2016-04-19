import numpy as np
from glm_links import link_g, link_p, link_l, get_y_hat
from iwls import iwls

def gauss_iwls(GLM):
    g_ey = GLM.y
    betas, w, v, n_iter = iwls(GLM.x, GLM.y, g_ey, link_g, GLM.family, GLM.offset,
            GLM.y_fix, GLM.sMatrix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['max_iter'])
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas, predy, v, w]

def poiss_iwls(GLM):     
    ey = GLM.y/GLM.offset
    g_ey = np.log(ey)
    betas, w, v, n_iter = iwls(GLM.x, GLM.y, g_ey, link_p, GLM.family, GLM.offset,
            GLM.y_fix, GLM.sMatrix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['maxIter'])
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas, predy, v, w]

def logit_iwls(GLM):
    ey = GLM.y/GLM.offset
    theta = 1.0 * np.sum(GLM.y)/GLM.n
    id_one = GLM.y == 1
    id_zero = GLM.y == 0
    g_ey = np.ones(shape=(GLM.n,1))
    g_ey[id_one] = np.log(theta/(1.0 - theta))
    g_ey[id_zero] = np.log((1.0 - theta)/theta)
    betas, w, v, n_iter = iwls(GLM.x, GLM.y, g_ey, link_l, GLM.family, GLM.offset,
            GLM.y_fix, GLM.sMatrix, GLM.fit_params['ini_betas'],
            GLM.fit_params['tol'], GLM.fit_params['max_iter'])
    GLM.fit_params['n_iter'] = n_iter
    predy = get_y_hat(GLM.family, v, GLM.offset, GLM.y_fix)
    return [betas, predy, v, w]
