# Author: Jing Yao
# July, 2013
# Univ. of St Andrews, Scotland, UK

# For diagnostics of Model estimation
# Some diagnostics for OLS model are from pysal

import numpy as np
from math import sqrt
from scipy import stats
from copy import copy

#--1 diagnostics for GWR Models----------------------------------------------------------------

def get_AICc_GWR(GWRMod):
    """
    Get AICc value
    
    Methods: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)
    using ML based sigma estimate
    
    Arguments:
        GWRMod: GWR model object
        
    Return:
        AICc value, float
    
    """
    n = GWRMod.nObs
    sigma2 = GWRMod.sigma2_ML
    v1 = GWRMod.tr_S
    aicc = -2.0*GWRMod.logll + 2.0*n*(v1 + 1.0)/(n-v1-2.0)#n*np.log(2.0*np.pi*sigma2) + n*((n+v1)/(n-2.0-v1)) #-2.0*GWRMod.logll + 2.0*n*(v1 + 1)/(n-v1-2)  
    
    return aicc#n*np.log(sigma2) + n*np.log(2.0*np.pi) + n*((n+v1)/(n-2.0-v1)) 

def get_AIC_GWR(GWRMod):
    """
    Get AIC value
    
    Methods: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)
    from Tomoki: AIC = -2L + 2.0 * (T_trace + 1.0);
    Note: for AIC, AICc and BIC of GWR, using (tr_S + 1) instead of k in the global model
    
    """
    #n = GWRMod.nObs
    #sigma2 = GWRMod.sigma2_ML#
    v1 = GWRMod.tr_S      
    aic = -2.0*GWRMod.logll + 2.0 * (v1 + 1) 
    
    return aic 
    
def get_BIC_GWR(GWRMod):
    """
    Get BIC value, BIC=-2log(L)+klog(n)
    
    Methods: p61, (2.34), Fotheringham, Brunsdon and Charlton (2002)
    from Tomoki:  BIC = -2L + (T_trace + 1.0) * log((double)N);
    """
    n = GWRMod.nObs
    v1 = GWRMod.tr_S 
    bic = -2.0 * GWRMod.logll + (v1+1) * np.log(n) 

    return bic
    
def get_CV_GWR(GWRMod):
    """
    Get CV value
    
    Methods: p60, (2.31) or p212 (9.4), Fotheringham, Brunsdon and Charlton (2002)
    Modification: sum of residual squared is divided by n according to GWR4 results
    
    Arguments:
        GWRMod: GWR model object     
            
    Return:
        cv: float, CV value       
    """
    aa = GWRMod.res/(1.0-GWRMod.influ)
    cv = np.sum(aa**2)/GWRMod.nObs
    
    return cv

#def logll_GWR(GWRMod):
    #"""
    #log likelihood??
    
    #Methods: p87 (4.2), Fotheringham, Brunsdon and Charlton (2002) 
    #from Tomoki: log-likelihood = -0.5 *(double)N * (log(ss / (double)N * 2.0 * PI) + 1.0);
    
    #"""
    #n = GWRMod.nObs
    ##u2 = GWRMod.res2
    #sigma2 = GWRMod.sigma2_ML   #sigma2_v1v2 which sigma should use
    #logll = -0.5*n*(np.log(2*np.pi*sigma2)+1) #np.log(-0.5*np.sum(u**2)/sigma2)
    
    #return logll

def r2_GWR(GWRMod):
    """
    Calculates the R^2 value for the GWR model. 
    
    Arguments:
        GWRMod      : GWR model object     
            
    Return:
        r2_result   : float
                      value of the coefficient of determination for the regression 

    Methods:    
    """ 
    tss = np.sum((GWRMod.y - GWRMod.y_mean)**2)
    r2 = 1.0 - GWRMod.res2/tss
    
    return r2

def ar2_GWR(GWRMod):
    """
    Calculates the adjusted R^2 value for GWR model. 
    
    PArguments:
        GWRMod      : GWR model object     
            
    Return:
        ar2_result  : float
                      value of the coefficient of determination for the regression 

    Methods:  
    
    """ 
    tss = np.sum((GWRMod.y - GWRMod.y_mean)**2)
    n = GWRMod.nObs       # (scalar) number of observations
    if GWRMod.tr_S >= GWRMod.tr_STS:		
	dof_res = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS
    else:
	dof_res = GWRMod.nObs-GWRMod.tr_S
    ar2_result =  1.0 - GWRMod.res2 / tss * ( n - 1.0) / (dof_res - 1.0)
    
    return ar2_result


#--End of diagnostics for GWR Models----------------------------------------------------------------

#--Global variable
get_criteria = {0: get_AICc_GWR, 1:get_AIC_GWR, 2:get_BIC_GWR, 3: get_CV_GWR} # bandwidth selection criteria

#--2 diagnostics for OLS Models---------------------------------------------------------------------
#def get_AICc_OLS(OLSMod):
    #"""
    #Get AICc value
    
    #Methods: AICc=AIC+2k(k+1)/(n-k-1)
    
    #Arguments:
        #GWRMod: GWR model object
        
    #Return:
        #AICc value, float
    
    #"""   
    #n = OLSMod.nObs       # (scalar) number of observations
    #k = OLSMod.nVars    # (scalar) number of ind. variables (including constant)
    #aicc = get_AIC_OLS(OLSMod) + 2.0*k*(k+1)/(n-k-1)
    
    #return aicc

#def get_AIC_OLS(OLSMod):
    #"""
    #Get AIC value, from pysal
    #The AIC = -2L+2K, where L is the log-likelihood and K is the number of parameters in the model
    
    #Methods: H. Akaike. 1974. A new look at the statistical identification
             #model. IEEE Transactions on Automatic Control, 19(6):716-723.
    
    #"""
    #n = OLSMod.nObs       # (scalar) number of observations
    #k = OLSMod.nVars    # (scalar) number of ind. variables (including constant)
    #res2 = OLSMod.res2   # (scalar) residual sum of squares
    #aic = 2*k + OLSMod.dev_res #- 2.0* OLSMod.logll #    
    
    #return aic
    
#def get_BIC_OLS(OLSMod):
    #"""
    #Calculates the Schwarz Information Criterion. BIC/SIC: SC = -2L+K*ln(N)

    #Parameters
    #----------
    #OLSMod          : regression object
                      #output instance from a regression model

    #Returns
    #-------
    #bic_result      : scalar
                      #value for Schwarz (Bayesian) Information Criterion of
                      #the regression. 

    #References
    #----------
    #.. [1] G. Schwarz. 1978. Estimating the dimension of a model. The
       #Annals of Statistics, pages 461-464. 
    #"""
    #n = OLSMod.nObs      # (scalar) number of observations
    #k = OLSMod.nVars   # (scalar) number of ind. variables (including constant)
    #utu = OLSMod.res2  # (scalar) residual sum of squares
    #bic_result = k*np.log(n) + OLSMod.dev_res #- 2.0 * OLSMod.logll  #n*(np.log((2*np.pi*utu)/n)+1)
    
    #return bic_result
    
def get_CV_OLS(OLSMod):
    """
    Get CV value
    
    prediction error: e_(i)=e_i/(1-h_ii), h is hat matrix, h = X(X'X)-1X'
    
    CV = e_(i)**2/n    
    
    Return:
        cv: float, CV value       
    """
    x_xtxi = np.dot(OLSMod.x, OLSMod.xtxi)
    h = np.dot(x_xtxi, OLSMod.x.T)
    h_diag = np.reshape(np.diag(h),(-1,1))    
    #n = OLSMod.nObs
    res = OLSMod.res/(1.0-h_diag)
    #res = np.zeros(shape=(OLSMod.nObs,1))
    #for i in range(n):
        #res[i] = OLSMod.res[i]/(1-h_diag[i])
    cv = OLSMod.res2/OLSMod.nObs
    
    return cv

def fstat_OLS(OLSMod):
    """
    Calculates the f-statistic and associated p-value of the regression.
    from pysal
    
    Parameters
    ----------
    OLSMod          : regression object
                      output instance from a regression model

    Returns
    ----------
    fs_result       : tuple
                      includes value of F statistic and associated p-value

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 
    """
    k = OLSMod.nVars            # (scalar) number of ind. vars (includes constant)
    n = OLSMod.nObs            # (scalar) number of observations
    utu = OLSMod.res2        # (scalar) residual sum of squares
    predy = OLSMod.y_pred    # (array) vector of predicted values (n x 1)
    mean_y = OLSMod.y_mean  # (scalar) mean of dependent observations
    Q = utu
    U = np.sum((predy-mean_y)**2)
    fStat = (U/(k-1))/(Q/(n-k))
    pValue = stats.f.sf(fStat,k-1,n-k)
    fs_result = (fStat, pValue)
    
    return fs_result

#def tstat_OLS(OLSMod, z_stat=False):
    #"""
    #Calculates the t-statistics (or z-statistics) and associated p-values.
    
    #Parameters
    #----------
    #OLSMod          : regression object
                      #output instance from a regression model
    #z_stat          : boolean
                      #If True run z-stat instead of t-stat
        
    #Returns
    #-------    
    #ts_result       : list of tuples
                      #each tuple includes value of t statistic (or z
                      #statistic) and associated p-value

    #References
    #----------

    #.. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       #Saddle River. 

    #""" 
    
    #k = OLSMod.nVars            # (scalar) number of ind. vars (includes constant)
    #n = OLSMod.nObs         # (scalar) number of observations
    #vm = OLSMod.var_Betas         # (array) coefficients of variance matrix (k x k)
    #betas = OLSMod.Betas   # (array) coefficients of the regressors (1 x k) 
    #variance = vm.diagonal()
    #tStat = betas[range(0,len(vm))].reshape(len(vm),)/ np.sqrt(variance)
    #ts_result = []
    #for t in tStat:
        #if z_stat:
            #ts_result.append((t, stats.norm.sf(abs(t))*2))
        #else:
            #ts_result.append((t, stats.t.sf(abs(t),n-k)*2))
            
    #return ts_result

def r2_OLS(OLSMod):
    """
    Calculates the R^2 value for the regression. 
    
    Parameters
    ----------
    OLSMod          : regression object
                      output instance from a regression model

    Returns
    ----------
    r2_result       : float
                      value of the coefficient of determination for the
                      regression 

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River.   
    """ 
    y = OLSMod.y               # (array) vector of dep observations (n x 1)
    mean_y = OLSMod.y_mean     # (scalar) mean of dep observations
    utu = OLSMod.res2           # (scalar) residual sum of squares
    ss_tot = ((y - mean_y) ** 2).sum(0)
    r2 = 1-utu/ss_tot
    r2_result = r2[0]
    return r2_result



def ar2_OLS(OLSMod):
    """
    Calculates the adjusted R^2 value for the regression. 
    
    Parameters
    ----------
    OLSMod          : regression object
                      output instance from a regression model   

    Returns
    ----------
    ar2_result      : float
                      value of R^2 adjusted for the number of explanatory
                      variables.

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 
    
    """ 
    k = OLSMod.nVars       # (scalar) number of ind. variables (includes constant)
    n = OLSMod.nObs       # (scalar) number of observations
    ar2_result =  1-(1-r2_OLS(OLSMod))*(n-1)/(n-k)
    return ar2_result



#def se_betas_OLS(OLSMod):
    #"""
    #Calculates the standard error of the regression coefficients.
    
    #Parameters
    #----------
    #OLSMod          : regression object
                      #output instance from a regression model

    #Returns
    #----------
    #se_result       : array
                      #includes standard errors of each coefficient (1 x k)

    #References
    #----------
    #.. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       #Saddle River. 
    
    #""" 
    #vm = OLSMod.var_Betas         # (array) coefficients of variance matrix (k x k)  
    #variance = vm.diagonal()
    #se_result = np.sqrt(variance)
    
    #return se_result



#def logll__OLS(OLSMod):
    #"""
    #Calculates the log-likelihood value for the regression. 
    
    #Parameters
    #----------
    #OLSMod             : regression object
                      #output instance from a regression model

    #Returns
    #-------
    #ll_result       : float
                      #value for the log-likelihood of the regression.

    #References
    #----------
    #.. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       #Saddle River. 
    #"""
    #n = OLSMod.nObs       # (scalar) number of observations
    #utu = OLSMod.res2   # (scalar) residual sum of squares
    #ll_result = -0.5*n*(np.log((2*np.pi*utu)/n)+1) #-0.5*(n*(np.log(2*np.pi))+n*np.log(utu/n)+(utu/(utu/n)))
    
    #return ll_result   


def ci_OLS(OLSMod):
    """
    Calculates the multicollinearity condition index according to Belsey,
    Kuh and Welsh (1980).

    Parameters
    ----------
    OLSMod          : regression object
                      output instance from a regression model

    Returns
    -------
    ci_result       : float
                      scalar value for the multicollinearity condition
                      index. 

    References
    ----------
    .. [1] D. Belsley, E. Kuh, and R. Welsch. 1980. Regression
       Diagnostics. New York: Wiley.

    """
    if hasattr(OLSMod, 'xtx'):
        xtx = OLSMod.xtx   # (array) k x k projection matrix (includes constant)
    elif hasattr(OLSMod, 'hth'):
        xtx = OLSMod.hth   # (array) k x k projection matrix (includes constant)
    diag = np.diagonal(xtx)
    scale = xtx/diag    
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_result = sqrt(max_eigval/min_eigval)
    
    return ci_result



#--End of diagnostics for OLS Models----------------------------------------------------------------

#--3 diagnostics for GLM Models---------------------------------------------------------------------
#def se_betas_GLM(GLMMod):
    #"""
    #Calculates the standard error of the regression coefficients.
    
    #Parameters
    #----------
    #GLMMod          : regression object
                      #output instance from a regression model

    #Returns
    #----------
    #se_result       : array
                      #includes standard errors of each coefficient (1 x k)    
    #""" 
    #vm = GLMMod.var_Betas         # (array) coefficients of variance matrix (k x k)  
    #variance = vm.diagonal()
    #se_result = np.sqrt(variance)
    
    #return se_result

def tstat_GLM(GLMMod, z_stat=False):
    """
    Calculates the t-statistics (or z-statistics) and associated p-values.
    
    Parameters
    ----------
    GLMMod          : regression object
                      output instance from a regression model
    z_stat          : boolean
                      If True run z-stat instead of t-stat
        
    Returns
    -------    
    ts_result       : list of tuples
                      each tuple includes value of t statistic (or z
                      statistic) and associated p-value

    References
    ----------

    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River. 

    """ 
    
    k = GLMMod.nVars            # (scalar) number of ind. vars (includes constant)
    n = GLMMod.nObs         # (scalar) number of observations
    #se_betas = np.reshape(np.sqrt(np.diag(GLMMod.var_Betas)),(-1,1))
    tStat = GLMMod.Betas/GLMMod.std_err#GLMMod.Betas/se_betas 
    ts_result = []
    for t in tStat:
        if z_stat:
            ts_result.append((t, stats.norm.sf(abs(t))*2))
        else:
            ts_result.append((t, stats.t.sf(abs(t),n-k)*2))
            
    return ts_result

def get_AICc_GLM(GLMMod):
    """
    Get AICc value
    
    Methods: AICc=AIC+2k(k+1)/(n-k-1)
    
    Arguments:
        GWRMod: GWR model object
        
    Return:
        AICc value, float
    
    """   
    n = GLMMod.nObs       # (scalar) number of observations
    k = GLMMod.nVars    # (scalar) number of ind. variables (including constant)
    aicc = get_AIC_GLM(GLMMod) + 2.0*k*(k+1)/(n-k-1)   
    
    return aicc

def get_AIC_GLM(GLMMod):
    """
    The AIC = -2L+2K, where L is the log-likelihood and K is the number of parameters in the model
    
       
    """   
    k = GLMMod.nVars  
    aic = GLMMod.dev_res + 2.0 * k   
    
    return aic
    
def get_BIC_GLM(GLMMod):
    """
    Calculates the Schwarz Information Criterion. BIC/SIC: SC = -2L+K*ln(N)

    Parameters
    ----------
    GLMMod          : regression object
                      output instance from a regression model

    Returns
    -------
    bic_result      : scalar
                      value for Schwarz (Bayesian) Information Criterion of
                      the regression. 
    """
    n = GLMMod.nObs      # (scalar) number of observations
    k = GLMMod.nVars   # (scalar) number of ind. variables (including constant)
    
    bic_result = GLMMod.dev_res + k*np.log(n) 
    
    return bic_result
    
#def dev_res_GLM(GLMMod):
    #"""
    #get residual deviance of GLM model
    #"""
    #if GLMMod.mType == 1:
	#id0 = GLMMod.y==0
	#id1 = GLMMod.y<>0
	
	#if np.sum(id1) == GLMMod.nObs:
	    #dev = 2.0 * np.sum(GLMMod.y * np.log(GLMMod.y/GLMMod.y_pred))
	#else:
	    #dev = 2.0 * (np.sum(GLMMod.y[id1] * np.log(GLMMod.y[id1]/GLMMod.y_pred[id1]))-np.sum(GLMMod.y[id0]-GLMMod.y_pred[id0]))   
	    
    #if GLMMod.mType == 2:
	#dev = 0.0
	#v = np.dot(GLMMod.x, GLMMod.Betas)
	#for i in range(GLMMod.nObs):
	    #if ((1.0 - GLMMod.y_pred[i]) < 1e-10): 
		#dev += -2.0 * (GLMMod.y[i] * v[i] + np.log(1e-10) )
	    #else: 
		#dev += -2.0 * (GLMMod.y[i] * v[i] + np.log(1.0 - GLMMod.y_pred[i]) )   
	#dev = dev[0]
       
    #return dev

def dev_mod_GLM(GLMMod):
    """
    get model deviance of GLM model: for Poisson and Logistic
    
    model type, 0: Gaussian, 1: Poisson, 2: Logistic
    """
    if GLMMod.mType == 1:
        ybar = np.sum(GLMMod.y)/np.sum(GLMMod.offset)    
        id0 = GLMMod.y==0
        id1 = GLMMod.y<>0
          
        if np.sum(id1) == GLMMod.nObs:    
            dev = 2.0 * np.sum(GLMMod.y * np.log(GLMMod.y/(ybar*GLMMod.offset)))  
        else:    
            dev = 2.0 * (np.sum(GLMMod.y[id1] * np.log(GLMMod.y[id1]/(ybar*GLMMod.offset[id1])))-np.sum(GLMMod.y[id0]-ybar*GLMMod.offset[id0]))  
            
    if GLMMod.mType == 2:
	ybar = np.sum(GLMMod.y) * 1.0/GLMMod.nObs
	v = np.log(ybar/(1.0 - ybar))
	dev = -2.0 * np.sum((GLMMod.y * v + np.log(1.0 - ybar)))
	
    return dev

#def pdev_GLM(GLMMod):
    #"""
    #percent of deviance
    #"""
    #dev_res = dev_res_GLM(GLMMod)
    #dev_full = dev_mod_GLM(GLMMod)
    
    #return 1.0 - dev_res/dev_full

#def logll_GLM(GLMMod):
    #"""
    #log likelihood of GLM model
    #"""
    #ll_result = np.sum(GLMMod.y * np.log(GLMMod.y_pred)-GLMMod.y_pred)
    
    #return ll_result

#--End of diagnostics for GLM Models----------------------------------------------------------------



#--4 diagnostics for GWGLM Models---------------------------------------------------------------------
#def dev_res_GWGLM(GLMMod):
    #"""
    #get residual deviance of GLM model
    #"""
    #dev = 0.0
    #if GLMMod.mType == 1:
	#for i in range(GLMMod.nObs):
	    #if (GLMMod.y[i] <> 0):
		#dev += 2 * GLMMod.y[i] * np.log(GLMMod.y[i]/GLMMod.y_pred[i])
	    #dev -= 2 * (GLMMod.y[i] - GLMMod.y_pred[i])
	#dev = dev[0]
	
	    
    #if GLMMod.mType == 2:
	#v = np.sum(GLMMod.x * GLMMod.Betas)
	##v1 = 1.0
	
	#for i in range(GLMMod.nObs):
	    #y_pred  = 1.0/(1.0 + np.exp(-v[i]))
	    #if ((1.0 - y_pred) < 1e-10): 
		#dev += -2.0 * (GLMMod.y[i] * v[i] + np.log(1e-10))
	    #else: 
		#dev += -2.0 * (GLMMod.y[i] * v[i] + np.log(1.0 - y_pred))   
	#dev = dev[0]
	
       
    #return dev

def get_AICc_GWGLM(GWGLMMod):
    """
    Get AICc value
    
    Methods: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)
    
    Arguments:
        GWRMod: GWR model object
        
    Return:
        AICc value, float
    
    """   
    n = GWGLMMod.nObs       # (scalar) number of observations
    k = GWGLMMod.tr_S     
    aicc = get_AIC_GWGLM(GWGLMMod) + 2 * k * (k + 1.0) / (n - k - 1.0) 
    
    return aicc

def get_AIC_GWGLM(GWGLMMod):
    """
    The AIC(G)=D(G) + 2K(G), where D and K denote the deviance and the effective number of parameters in the model
    with bandwidth G, respectively.
    
    Methods: Nakaya et al. (2005): p2703, (35)
    
    """   
    k = GWGLMMod.tr_S  
    aic = GWGLMMod.dev_res + 2.0 * k   
    
    return aic

def get_BIC_GWGLM(GWGLMMod):
    """
    Calculates the Schwarz Information Criterion. BIC/SIC: SC = dev + T_trace * log((double)N);

    Parameters
    ----------
    GLMMod          : regression object
                      output instance from a regression model

    Returns
    -------
    bic_result      : scalar
                      value for Schwarz (Bayesian) Information Criterion of
                      the regression. 
    """
    n = GWGLMMod.nObs      # (scalar) number of observations
    k = GWGLMMod.tr_S   
    
    bic_result = GWGLMMod.dev_res + k*np.log(n) 
    
    return bic_result


#For Gaussian model, calculate local R2


    


#--End of diagnostics for GWGLM Models----------------------------------------------------------------


    
