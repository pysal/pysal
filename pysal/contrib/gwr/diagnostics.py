"""
Diagnostics for gwr estimations.
"""
__author__ = ""

import numpy as np

__all__ = [
    "f_stat", "t_stat", "r2", "ar2", "se_betas", "log_likelihood", "akaike", "schwarz",
    "condition_index", "jarque_bera", "breusch_pagan", "white", "koenker_bassett", "vif", "likratiotest"]


def get_AICc_GWR(GWR):
    """
    Get AICc value

    Methods: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    using ML based sigma estimate

    Arguments:
        GWR: GWR model object

    Return:
        AICc value, float

    """
    n = GWR.n
    v1 = GWR.tr_S
    aicc = -2.0*GWR.logll + 2.0*n*(v1 + 1.0)/(n-v1-2.0)

    return aicc

def get_AIC_GWR(GWR):
    """
    Get AIC value

    Methods: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)
    from Tomoki: AIC = -2L + 2.0 * (T_trace + 1.0);
    Note: for AIC, AICc and BIC of GWR, using (tr_S + 1) instead of k in the global model

    """
    v1 = GWR.tr_S
    aic = -2.0*GWR.logll + 2.0 * (v1 + 1)

    return aic

def get_BIC_GWR(GWR):
    """
    Get BIC value, BIC=-2log(L)+klog(n)

    Methods: p61, (2.34), Fotheringham, Brunsdon and Charlton (2002)
    from Tomoki:  BIC = -2L + (T_trace + 1.0) * log((double)N);
    """
    n = GWR.n
    v1 = GWR.tr_S
    bic = -2.0 * GWR.logll + (v1+1) * np.log(n)

    return bic

def get_CV_GWR(GWRMod):
    """
    Get CV value

    Methods: p60, (2.31) or p212 (9.4), Fotheringham, Brunsdon and Charlton (2002)
    Modification: sum of residual squared is divided by n according to GWR4 results

    Arguments:
        GWR: GWR model object

    Return:
        cv: float, CV value
    """
    aa = GWR.res/(1.0-GWR.influ)
    cv = np.sum(aa**2)/GWR.n

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

    PArguments:
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


