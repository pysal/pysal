"""
Diagnostics for estimated gwr modesl
"""
__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
from pysal.contrib.glm.family import Gaussian, Poisson, Binomial

def get_AICc(gwr):
    """
    Get AICc value
    
    Gaussian: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    
    GWGLM: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)

    """
    n = gwr.n
    k = gwr.tr_S
    if isinstance(gwr.family, Gaussian):
        aicc = -2.0*gwr.llf + 2.0*n*(k + 1.0)/(n-k-2.0)  
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aicc = get_AIC(gwr) + 2.0 * k * (k+1.0) / (n - k - 1.0) 
    return aicc

def get_AIC(gwr):
    """
    Get AIC calue

    Gaussian: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)

    GWGLM:  AIC(G)=D(G) + 2K(G), where D and K denote the deviance and the effective
    number of parameters in the model with bandwidth G, respectively.
    
    """   
    k = gwr.tr_S
    #deviance = -2*log-likelihood
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        aic = -2.0 * gwr.llf + 2.0 * (k+1)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aic = np.sum(gwr.family.resid_dev(y, mu)**2) + 2.0 * k
    return aic 

def get_BIC(gwr):
    """
    Get BIC value

    Gaussian: p61 (2.34), Fotheringham, Brunsdon and Charlton (2002)
    BIC = -2log(L)+klog(n)

    GWGLM: BIC = dev + tr_S * log(n)

    """
    n = gwr.n      # (scalar) number of observations
    k = gwr.tr_S  
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        bic = -2.0 * gwr.llf + (k+1) * np.log(n) 
    elif isinstance(gwr.family, (Poisson, Binomial)):
        bic = np.sum(gwr.family.resid_dev(y, mu)**2) + k * np.log(n)
    return bic

def get_CV(gwr):
    """
    Get CV value

    Gaussian only

    Methods: p60, (2.31) or p212 (9.4)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    Modification: sum of residual squared is divided by n according to GWR4 results

    """
    aa = gwr.resid_response.reshape((-1,1))/(1.0-gwr.influ)
    cv = np.sum(aa**2)/gwr.n
    return cv

