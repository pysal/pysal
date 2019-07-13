from __future__ import division
import scipy.stats as stats
import numpy as np
from warnings import warn
from ..steps import metropolis

def logp_phi(state, phi):
    """
    This is the log of the probability distribution in equation 7 of the
    appendix of the Wheeler and Calder (2010) paper on svcp
    """
    if phi < 0:
        return np.array([-np.inf])
    st = state
    # NOTE: I'm exploiting the following property for two square
    # matrices, H,T, of shapes n x n and p x p, respectively:
    # log(det(H kron T)) = log(det(H)^p * log(det(T)^n))
    # = log(det(H))*p + log(det(T))*n

    sgnH, logdetH = np.linalg.slogdet(st.H)
    sgnT, logdetT = np.linalg.slogdet(st.T)
    logdetH *= sgnH
    logdetT *= sgnT
    if any([x not in (-1,1) for x in [sgnH, sgnT]]):
        warn('Catastrophic loss of precision in np.linalg.slogdet of np.kron(st.H, st.T)')
    logdet = logdetH * st.p + logdetT * st.N
    Bmu = st.Betas - st.tiled_Mus
    kronHT_inv = st.kronHiTi #since inv of kronecker is kronecker of invs
    normal_kernel = np.dot(Bmu.T, np.dot(kronHT_inv, Bmu)) * -.5
    gamma_kernel = np.log(phi)*(st.Phi_shape0 - 1) + -1*st.Phi_rate0*phi
    return -.5*logdet + normal_kernel + gamma_kernel

def sample_phi(SVCP):
    """
    Sample phi, conditional on the state contained in the SVCP sampler

    Parameters
    ----------
    SVCP    :   sampler
                the execution context in which phi is to be sampled

    Returns
    --------
    None. works by sampling in place on SVCP. It updates:
    configs.phi.accepted OR configs.phi.rejected
    configs.phi.jump if tuning
    configs.phi.tuning if ending tuning
    state.Phi
    """
    state = SVCP.state
    cfg = SVCP.configs
    current = state.Phi
    try:
        #special proposals can be stored in configs
        proposal = cfg.Phi.proposal
    except KeyError:
        #if we don't have a proposal, take it to be a normal proposal
        proposal = stats.normal
        # and short circuit this assignment for later
        cfg.Phi.proposal = proposal
    new_val, accepted = metropolis(state, current, proposal,
                                   logp_phi,cfg.Phi.jump)

    if accepted:
        cfg.Phi.accepted += 1
    else:
        cfg.Phi.rejected += 1
    if cfg.tuning:
        acc = cfg.Phi.accepted
        rej = cfg.Phi.rejected
        ar = acc / (acc + rej)
        if ar < cfg.Phi.ar_low:
            cfg.Phi.jump *= cfg.Phi.adapt_step
        elif ar > cfg.Phi.ar_hi:
            cfg.Phi.jump /= cfg.Phi.adapt_step
        if SVCP.cycles >= cfg.Phi.max_tuning:
            cfg.tuning = False
    return new_val
