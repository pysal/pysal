import numpy as np
import scipy.linalg as scla
from ...utils import splogdet
from pysal.spreg.utils import spdot

def logp_rho_prec(state, val):
    """
    This computes the logp of the spatial parameter using the precision, rather than the covariance. This results in fewer matrix operations in the case of a SE formulation, but not in an SMA formulation.
    """
    st = state

    #must truncate in logp otherwise sampling gets unstable
    if (val < st.Rho_min) or (val > st.Rho_max):
        return np.array([-np.inf])

    PsiRhoi = st.Psi_1i(val, st.W, sparse=True)
    logdet = splogdet(PsiRhoi)

    eta = st.Y - st.XBetas - st.DeltaAlphas
    kernel = spdot(spdot(eta.T, PsiRhoi), eta) / st.Sigma2

    return .5*logdet -.5 * kernel + st.Log_Rho0(val) #since precision, no negative on ld


def logp_lambda_prec(state, val):
    """
    The logp for upper level spatial parameters in this case has the same form
    as a multivariate normal distribution, sampled over the variance matrix,
    rather than over Y.
    """
    st = state

    #must truncate
    if (val < st.Lambda_min) or (val > st.Lambda_max):
        return np.array([-np.inf])

    PsiLambdai = st.Psi_2i(val, st.M)
    logdet = splogdet(PsiLambdai)

    kernel = spdot(spdot(st.Alphas.T, PsiLambdai), st.Alphas) / st.Tau2

    return .5*logdet - .5*kernel + st.Log_Lambda0(val)
