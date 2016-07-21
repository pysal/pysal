"""
Diagnostics for gwr estimations.
"""
__author__ = ""

import numpy as np

def get_AICc_GWR(GWR):
    """
    Get AICc value

    Methods: p61, (2.33)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    using ML based sigma estimate

    Arguments:
        GWR: GWR model object

    Return:
        AICc value, float

    """
    n = GWR.n
    v1 = GWR.tr_S
    sig = GWR.sigma2_ML
    #aicc = -2.0*GWR.logll + 2.0*n*(v1 + 1.0)/(n-v1-2.0)
    aicc = 2.0*n*np.log(GWR.std_err) + n*np.log(2.0*np.pi) + n*((n + v1)/(n - 2.0 - v1))
    #aicc = n*np.log(sig) + n*np.log(2.0*np.pi) + n*((n+v1)/(n-2.0-v1))
    #aicc = GWR.dev_u + 2.0*v1
    print aicc
    return aicc

def get_AIC_GWR(GWR):
    """
    Get AIC value

    Methods: p96, (4.22)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    from Tomoki: AIC = -2L + 2.0 * (T_trace + 1.0);
    Note: for AIC, AICc and BIC of GWR, using (tr_S + 1) instead of k in the global model

    """
    v1 = GWR.tr_S
    aic = -2.0*GWR.logll + 2.0 * (v1 + 1)

    return aic

def get_BIC_GWR(GWR):
    """
    Get BIC value, BIC=-2log(L)+klog(n)

    Methods: p61, (2.34)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    from Tomoki:  BIC = -2L + (T_trace + 1.0) * log((double)N);
    """
    n = GWR.n
    v1 = GWR.tr_S
    bic = -2.0 * GWR.logll + (v1+1) * np.log(n)

    return bic

def get_CV_GWR(GWR):
    """
    Get CV value

    Methods: p60, (2.31) or p212 (9.4)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    Modification: sum of residual squared is divided by n according to GWR4 results

    Arguments:
        GWR: GWR model object

    Return:
        cv: float, CV value
    """
    aa = GWR.u/(1.0-GWR.influ)
    cv = np.sum(aa**2)/GWR.n
    print cv
    return cv

def r2_GWR(GWR):
    """
    Calculates the R^2 value for the GWR model.

    Arguments:
        GWR         : GWR model object

    Return:
        r2_result   : float
                      value of the coefficient of determination for the regression

    Methods:
    """
    tss = np.sum((GWR.y - GWR.y_mean)**2)
    r2 = 1.0 - GWR.utu/tss

    return r2

def ar2_GWR(GWR):
    """
    Calculates the adjusted R^2 value for GWR model.

    Arguments:
        GWR      : GWR model object

    Return:
        ar2_result  : float
                      value of the coefficient of determination for the regression

    Methods:

    """
    tss = np.sum((GWR.y - GWR.y_mean)**2)
    n = GWR.n       # (scalar) number of observations
    if GWR.tr_S >= GWR.tr_STS:
	dof_res = GWR.n-2.0*GWR.tr_S+GWR.tr_STS
    else:
	dof_res = GWR.n-GWR.tr_S
    ar2_result =  1.0 - GWR.utu / tss * ( n - 1.0) / (dof_res - 1.0)

    return ar2_result

def get_AICc_GLM(GLMMod):
    """
    Get AICc value
    
    Methods: AICc=AIC+2k(k+1)/(n-k-1)
    
    Arguments:
        GWRMod: GWR model object
        
    Return:
        AICc value, float
    
    """   
    n = GLMMod.n      # (scalar) number of observations
    k = GLMMod.k+1    # (scalar) number of ind. variables (including constant)
    aicc = get_AIC_GLM(GLMMod) + 2.0*k*(k+1)/(n-k-1)   
    
    return aicc



def get_AIC_GLM(GLMMod):
    """
    The AIC = -2L+2K, where L is the log-likelihood and K is the number of parameters in the model
    
       
    """   
    k = GLMMod.k+1  
    print GLMMod.dev_u
    aic = -2.0 * GLMMod.dev_u + 2.0 * k   
    
    return aic
