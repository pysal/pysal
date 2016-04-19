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
    #sigma2 = GWR.sigma2_ML
    v1 = GWR.tr_S
    aicc = -2.0*GWR.logll + 2.0*n*(v1 + 1.0)/(n-v1-2.0)  
    
    return aicc 
