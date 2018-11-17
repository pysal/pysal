from __future__ import division
import numpy as np
import numpy.linalg as la
from ...utils import splogdet, spsolve
from ...steps import metropolis
from pysal.spreg.utils import spdot

#############################
# SPATIAL SAMPLE METHODS    #
#############################

def logp_rho_cov(state, val):
    """
    The logp for lower-level spatial parameters in this case has the same
    form as a multivariate normal distribution, sampled over the variance matrix, rather than over y.
    """
    st = state
    
    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])
    
    PsiRho = st.Psi_1(val, st.W)
    logdet = splogdet(PsiRho)
    
    eta = st.Y - st.XBetas - st.DeltaAlphas
    kernel = spdot(eta.T, spsolve(PsiRho, eta)) / st.Sigma2

    return -.5*logdet -.5 * kernel + st.Log_Rho0(val)

def logp_lambda_cov(state, val):
    """
    The logp for upper level spatial parameters in this case has the same form
    as a multivariate normal distribution, sampled over the variance matrix,
    rather than over Y.
    """
    st = state

    #must truncate
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])

    PsiLambda = st.Psi_2(val, st.M)

    logdet = splogdet(PsiLambda)

    kernel = spdot(st.Alphas.T, spsolve(PsiLambda, st.Alphas)) / st.Tau2

    return -.5*logdet - .5*kernel + st.Log_Lambda0(val)

def logp_lambda_prec(state, val):
    """
    Compute the log likelihood of the upper-level spatial correlation parameter using 
    sparse operations and the precision matrix, rather than the covariance matrix. 
    """
    st = state

    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])

    PsiLambdai = st.Psi_2i(val, st.M, sparse=True)
    logdet = -splogdet(PsiLambdai) #negative because precision

    kernel = spdot(spdot(st.Alphas.T, PsiLambdai), st.Alphas) / st.Tau2

    return -.5 * logdet - .5 * kernel + st.Log_Lambda0(val)

def logp_rho_prec(state, val):
    """
    Compute the log likelihood of the lower-level spatial correlation parameter using
    sparse operations and the precision matrix, rather than the covariance matrix. 
    """
    st = state

    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])

    PsiRhoi = st.Psi_1i(val, st.W, sparse=True)
    logdet = -splogdet(PsiRhoi)

    eta = st.Y - st.XBetas - st.DeltaAlphas

    kernel = spdot(spdot(eta.T, PsiRhoi), eta) / st.Sigma2

    return -.5 * logdet - .5 * kernel + st.Log_Rho0(val)


