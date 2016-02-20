# Author: Jing Yao
# July, 2013
# Univ. of St Andrews, Scotland, UK

# For output summary

import Diagnostics
import numpy as np
from datetime import datetime
from scipy.stats.mstats import mquantiles
from scipy import stats


#def GWR(GWRMod):
    #"""
    #get diagnostics for GWR model
    #"""
    ##GWRMod.logll = Diagnostics.logll_GWR(GWRMod) 
    #GWRMod.aic = Diagnostics.get_AIC_GWR(GWRMod)
    #GWRMod.aicc = Diagnostics.get_AICc_GWR(GWRMod)
    #GWRMod.bic = Diagnostics.get_BIC_GWR(GWRMod) 
    #GWRMod.cv = Diagnostics.get_CV_GWR(GWRMod)
    #GWRMod.R2 = Diagnostics.r2_GWR(GWRMod)
    #GWRMod.R2_adj = Diagnostics.ar2_GWR(GWRMod)
    #summary_GWR(GWRMod)
    
#def summary_GWR(GWRMod):
    #"""
    #output summary in string
    #"""
    #dic_sum = {}
    
    #dic_sum['Caption'] = "%s\n" % ('Summary: Geographically Weighted Regression')
    #dic_sum['BeginT'] = "%-21s: %s %s\n" % ('Program started at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    #dic_sum['DataSource'] = "%s %s\n" % ('Data filename:', GWRMod.fle_name)
    #dic_sum['ModSettings'] = "%s\n" % ('Model settings:')
    #dic_sum['ModOptions'] = "%s\n" % ('Modelling options:')
    #dic_sum['VarSettings'] = "%s\n" % ('Variable settings:')
    #dic_sum['GlobResult'] = '' 
    #dic_sum['GlobResult_diag'] = ''
    #dic_sum['GlobResult_esti'] = ''  
    #dic_sum['GWRResult'] = "%s\n" %('GWR (Geographically weighted regression) result')
    #dic_sum['GWR_band'] = "%s\n" %('GWR (Geographically weighted regression) bandwidth selection')
    #dic_sum['GWR_diag'] = "%s\n" % ('Diagnostic information') 
    #dic_sum['GWR_esti'] = "%s\n" % ('<< Geographically varying (Local) coefficients >>')
    #dic_sum['GWR_anova'] = "%s\n" %('GWR ANOVA Table') 
    #dic_sum['EndT'] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    
    #dic_sum['Caption'] += '-' * 75 + '\n'
    #dic_sum['Caption'] += "\n"
    
    #dic_sum['DataSource'] += "%-45s %d\n" % ('Number of observations:', GWRMod.nObs) 
    #dic_sum['DataSource'] += "%-45s %d\n" % ('Number of Variables:', GWRMod.nVars) 
    #dic_sum['DataSource'] += "\n"   
    
    #dic_sum['ModSettings'] += '-' * 75 + '\n'
    #dic_sum['ModSettings'] += "%-45s %s\n" % ('Model type:', GWRMod.mName)
    #dic_sum['ModSettings'] += "%-45s %s\n" % ('Geographic kernel:', GWRMod.kernel.wName)
    #dic_sum['ModSettings'] += "\n"
    
    #dic_sum['ModOptions'] += '-' * 75 + '\n'
    #dic_sum['ModOptions'] += "\n"  
       
    #dic_sum['VarSettings'] += '-' * 75 + '\n'
    #dic_sum['VarSettings'] += "%-60s %s\n" % ('Dependent variable:', GWRMod.y_name)
    #for xVar in GWRMod.x_name:
        #dic_sum['VarSettings'] += "%-60s %s\n" % ('Independent variable with varying (Local) coefficient:', xVar)
    #dic_sum['VarSettings'] += "\n"   
    
    #dic_sum['GWRResult'] += '-' * 75 + '\n'
    #dic_sum['GWRResult'] += "%s\n" %('Geographic ranges')
    #dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('Coordinate', 'Min', 'Max', 'Range')
    #dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('-'*20, '-'*20, '-'*20, '-'*20)
    #arr_coords = np.array(GWRMod.kernel.coords.values())
    #arr_X = arr_coords[:,0]
    #arr_Y = arr_coords[:,1]
    #min_x = min(arr_X)
    #max_x = max(arr_X)
    #min_y = min(arr_Y)
    #max_y = max(arr_Y)
    #dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('X-coord', min_x, max_x, max_x-min_x)
    #dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('Y-coord', min_y, max_y, max_y-min_y)
    #dic_sum['GWRResult'] += "\n"
    
    #dic_sum['GWR_band'] += '-' * 75 + '\n'
    #dic_sum['GWR_band'] += "%-37s %20.6f\n" % ('Bandwidth size:', GWRMod.kernel.band)
    #dic_sum['GWR_band'] += "\n"
    
    #dic_sum['GWR_diag'] += '-' * 75 + '\n'
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', np.sum(GWRMod.res**2))
    #dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (model: trace(S)):', GWRMod.tr_S)
    #dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (variance: trace(S' + "'" + 'S))', GWRMod.tr_STS)
    #dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (model: n - trace(S)):', GWRMod.nObs-GWRMod.tr_S)
    #dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (residual: n - 2trace(S) + trace(S' + "'" + 'S)):', GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS)
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('ML based sigma estimate:', np.sqrt(GWRMod.sigma2_ML))
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Unbiased sigma estimate:', np.sqrt(GWRMod.sigma2_v1v2))
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', -2.0*GWRMod.logll) 
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GWRMod.aic)
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('AICc:', GWRMod.aicc)
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('BIC:', GWRMod.bic) 
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('CV:', GWRMod.cv)
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('R square:', GWRMod.R2)
    #dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GWRMod.R2_adj)
    #dic_sum['GWR_diag'] += "\n"
    
    #dic_sum['GWR_esti'] += "%s\n\n" % ('Summary statistics for varying (Local) coefficients')
    #dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Mean' ,'STD')
    #dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    #for i in range(GWRMod.nVars):
        #dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name[i], np.mean(GWRMod.Betas[:,i]) ,np.std(GWRMod.Betas[:,i]))
    #dic_sum['GWR_esti'] += "\n"
    #dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Min' ,'Max', 'Range')
    #dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    #for i in range(GWRMod.nVars):
        #dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name[i], np.min(GWRMod.Betas[:,i]) ,np.max(GWRMod.Betas[:,i]), np.max(GWRMod.Betas[:,i])-np.min(GWRMod.Betas[:,i]))
    #dic_sum['GWR_esti'] += "\n"    
    #dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Lwr Quartile' ,'Median', 'Upr Quartile')              
    #dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    #for i in range(GWRMod.nVars):
        #quan = mquantiles(GWRMod.Betas[:,i])
        #dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name[i], quan[0],np.median(GWRMod.Betas[:,i]), quan[2])    
    #dic_sum['GWR_esti'] += "\n"    
    #dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Interquartile R' ,'Robust STD')              
    #dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    #for i in range(GWRMod.nVars):
        #quan = mquantiles(GWRMod.Betas[:,i])
        #dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name[i], quan[2]-quan[0], (quan[2]-quan[0])/1.349)      
    #dic_sum['GWR_esti'] += "\n"    
    #dic_sum['GWR_esti'] += "%s\n" % ('(Note: Robust STD is given by (interquartile range / 1.349) )')
    #dic_sum['GWR_esti'] += "\n"
    
    #dic_sum['GWR_anova'] += '-' * 75 + '\n'
    #dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('Source', 'SS', 'DF', 'MS', 'F')
    #dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('-'*20, '-'*20, '-'*20, '-'*20, '-'*20)
    #df_ols = GWRMod.nObs-GWRMod.nVars
    #df_gwr = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS
    #if hasattr(GWRMod, 'GLM'):        
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f\n" % ('Global Residuals', GWRMod.GLM.res2, df_ols)
        #ms_imp = (GWRMod.GLM.res2-GWRMod.res2)/(df_ols-df_gwr)
        #ms_gwr = GWRMod.res2/df_gwr
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR Improvement', GWRMod.GLM.res2-GWRMod.res2, df_ols-df_gwr, ms_imp)
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % ('GWR Residuals', GWRMod.res2, df_gwr, ms_gwr, ms_imp/ms_gwr)
    #dic_sum['GWR_anova'] += "\n"
    
    #sumStr = '' 
    #sumStr += dic_sum['Caption']
    #sumStr += dic_sum['BeginT']
    #sumStr += dic_sum['DataSource']
    #sumStr += dic_sum['ModSettings']    
    #sumStr += dic_sum['ModOptions']
    #sumStr += dic_sum['VarSettings']    
    #sumStr += dic_sum['GlobResult']
    #sumStr += dic_sum['GlobResult_diag']    
    #sumStr += dic_sum['GlobResult_esti']
    #sumStr += dic_sum['GWRResult']     
    #sumStr += dic_sum['GWR_band']
    #sumStr += dic_sum['GWR_diag']    
    #sumStr += dic_sum['GWR_esti']
    #sumStr += dic_sum['GWR_anova']    
    #sumStr += dic_sum['EndT']
        
    
    #GWRMod.summary = sumStr
    
#def OLS(OLSMod):
    #"""
    #get diagnostics for OLS model
    #"""
    #OLSMod.t_stat = Diagnostics.tstat_OLS(OLSMod)
    #OLSMod.f_stat = Diagnostics.fstat_OLS(OLSMod)
    #OLSMod.aic = Diagnostics.get_AIC_OLS(OLSMod)
    #OLSMod.aicc = Diagnostics.get_AICc_OLS(OLSMod) #not finish yet
    #OLSMod.r2 = Diagnostics.r2_OLS(OLSMod)
    #OLSMod.ar2 = Diagnostics.ar2_OLS(OLSMod)
    #OLSMod.std_err = Diagnostics.se_betas_OLS(OLSMod)
    ##OLSMod.logll = Diagnostics.logll__OLS(OLSMod)
    #OLSMod.bic = Diagnostics.get_BIC_OLS(OLSMod)
    #OLSMod.cv = Diagnostics.get_CV_OLS(OLSMod) 
    #OLSMod.sig2ML = OLSMod.sigma2_n 
    #OLSMod.mulColli = Diagnostics.ci_OLS(OLSMod)
    
    #summary_OLS(OLSMod)
    
#def summary_OLS(OLSMod):
    #"""
    #output summary in string
    #"""
    #dic_sum = {}
    
    #dic_sum['Caption'] = "%s\n" % ('Summary: Ordinary Least Squares Estimation')
    #dic_sum['BeginT'] = "%-21s: %s %s\n" % ('Program started at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    #dic_sum['DataSource'] = "%s %s\n" % ('Data filename:', OLSMod.fle_name)
    #dic_sum['ModSettings'] = "%s\n" % ('Model settings:')
    #dic_sum['ModOptions'] = ''
    #dic_sum['VarSettings'] = ''
    #dic_sum['OLSResult'] = "%s\n"  % ('Global regression result')
    #dic_sum['OLS_diag'] = "%s\n" % ('< Diagnostic information >')
    #dic_sum['OLS_esti'] = ''      
    #dic_sum['EndT'] = "%-21s: %s %s\n" % ('Program terminated at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    
    #dic_sum['Caption'] += '-' * 75 + '\n'
    #dic_sum['Caption'] += "\n"
    
    #dic_sum['DataSource'] += "%-45s %d\n" % ('Number of observations:', OLSMod.nObs) 
    #dic_sum['DataSource'] += "%-45s %d\n" % ('Number of Variables:', OLSMod.nVars) 
    #dic_sum['DataSource'] += "\n"
    
    #dic_sum['ModSettings'] += '-' * 75 + '\n'
    #dic_sum['ModSettings'] += "%-45s %s\n" % ('Model type:', OLSMod.mName)
    #dic_sum['ModSettings'] += "\n"
    
    #dic_sum['VarSettings'] += '-' * 75 + '\n'
    #dic_sum['VarSettings'] += "%-45s %12s\n" % ('Dependent variable:', OLSMod.y_name)
    #dic_sum['VarSettings'] += "\n"        
    
    #dic_sum['OLS_diag'] += '-' * 75 + '\n'
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', OLSMod.res2)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('ML based global sigma estimate:', np.sqrt(OLSMod.sig2ML))
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('Unbiased global sigma estimate:', np.sqrt(OLSMod.sigma2_nk))
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', OLSMod.dev_res)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', OLSMod.aic)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('AICc:', OLSMod.aicc)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('BIC/MDL:', OLSMod.bic)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('CV:', OLSMod.cv)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('R square:', OLSMod.r2)
    #dic_sum['OLS_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', OLSMod.ar2) 
    #dic_sum['OLS_diag'] += "\n"
    
    
    #dic_sum['OLS_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'t(Est/SE)', 'p-value')
    #dic_sum['OLS_esti'] += "---------------------------------------------------------------------------------------------------------\n"
    #for i in range(OLSMod.nVars):
        #dic_sum['OLS_esti'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (OLSMod.x_name[i], OLSMod.Betas[i], OLSMod.std_err[i] ,OLSMod.t_stat[i][0], OLSMod.t_stat[i][1])
    
    #dic_sum['OLS_esti'] += "\n"
    
    #sumStr = '' 
    #sumStr += dic_sum['Caption']
    #sumStr += dic_sum['BeginT']
    #sumStr += dic_sum['DataSource']
    #sumStr += dic_sum['ModSettings']    
    #sumStr += dic_sum['ModOptions']
    #sumStr += dic_sum['VarSettings']    
    #sumStr += dic_sum['OLSResult']
    #sumStr += dic_sum['OLS_diag']    
    #sumStr += dic_sum['OLS_esti']
    #sumStr += dic_sum['EndT']
        
    
    #OLSMod.summary = sumStr
    
    
def GLM(GLMMod):
    """
    get diagnostics for GLM model
    """
    if GLMMod.mType == 0:
        GLMMod.t_stat = Diagnostics.tstat_GLM(GLMMod)
        GLMMod.f_stat = Diagnostics.fstat_OLS(GLMMod)        
        GLMMod.r2 = Diagnostics.r2_OLS(GLMMod)
        GLMMod.ar2 = Diagnostics.ar2_OLS(GLMMod)       
        GLMMod.cv = Diagnostics.get_CV_OLS(GLMMod) 
        GLMMod.sig2ML = GLMMod.sigma2_n 
        GLMMod.mulColli = Diagnostics.ci_OLS(GLMMod)
    
        #summary_OLS(GLMMod)
    else:        
        GLMMod.dev_null = Diagnostics.dev_mod_GLM(GLMMod)
        GLMMod.pdev = 1.0 - GLMMod.dev_res/GLMMod.dev_null  
        GLMMod.t_stat = Diagnostics.tstat_GLM(GLMMod, True) 
        
    #GLMMod.std_err = Diagnostics.se_betas_GLM(GLMMod)
    GLMMod.aic = Diagnostics.get_AIC_GLM(GLMMod)
    GLMMod.aicc = Diagnostics.get_AICc_GLM(GLMMod) 
    GLMMod.bic = Diagnostics.get_BIC_GLM(GLMMod) 
    
    summary_GLM(GLMMod)
    
def summary_GLM(GLMMod):
    """
    output summary in string
    """
    dic_sum = {}
    
    if GLMMod.mType == 0:
        dic_sum['Caption'] = "%s\n" % ('Summary: Ordinary Least Squares Estimation')
    else:
        dic_sum['Caption'] = "%s\n" % ('Summary: Generalised linear model (GLM)')
    dic_sum['BeginT'] = "%-21s: %s %s\n" % ('Program started at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    dic_sum['DataSource'] = "%s %s\n" % ('Data filename:', GLMMod.fle_name)
    dic_sum['ModSettings'] = "%s\n" % ('Model settings:')
    dic_sum['ModOptions'] = ''
    dic_sum['VarSettings'] = ''
    dic_sum['GLMResult'] = "%s\n"  % ('Global regression result')
    dic_sum['GLM_diag'] = "%s\n" % ('< Diagnostic information >')
    dic_sum['GLM_esti'] = ''      
    dic_sum['EndT'] = "%-21s: %s %s\n" % ('Program terminated at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    
    dic_sum['Caption'] += '-' * 75 + '\n'
    dic_sum['Caption'] += "\n"
    
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of observations:', GLMMod.nObs) 
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of Variables:', GLMMod.nVars) 
    dic_sum['DataSource'] += "\n"
    
    dic_sum['ModSettings'] += '-' * 75 + '\n'
    dic_sum['ModSettings'] += "%-45s %s\n" % ('Model type:', GLMMod.mName)
    dic_sum['ModSettings'] += "\n"
    
    dic_sum['VarSettings'] += '-' * 75 + '\n'
    dic_sum['VarSettings'] += "%-45s %12s\n" % ('Dependent variable:', GLMMod.y_name)
    if GLMMod.mType == 1:
        dic_sum['VarSettings'] += "%-45s %12s\n" % ('Offset variable:', GLMMod.y_off_name)
    dic_sum['VarSettings'] += "\n"        
    
    dic_sum['GLM_diag'] += '-' * 75 + '\n'
    if GLMMod.mType == 0:
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', GLMMod.res2)
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('ML based global sigma estimate:', np.sqrt(GLMMod.sig2ML))
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Unbiased global sigma estimate:', np.sqrt(GLMMod.sigma2_nk))
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', GLMMod.dev_res)
    dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GLMMod.aic)
    dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('AICc:', GLMMod.aicc)
    dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('BIC/MDL:', GLMMod.bic)
    if GLMMod.mType == 0:
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('CV:', GLMMod.cv)
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('R square:', GLMMod.r2)
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GLMMod.ar2) 
    else:
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Null deviance:', GLMMod.dev_null)
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Residual deviance:', GLMMod.dev_res)
        dic_sum['GLM_diag'] += "%-45s %12.6f\n" % ('Percent deviance explained:', GLMMod.pdev)
    dic_sum['GLM_diag'] += "\n"
    
    
    if GLMMod.mType == 0:
        dic_sum['GLM_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'t(Est/SE)', 'p-value')
    else:
        dic_sum['GLM_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'z(Est/SE)', 'p-value')
    dic_sum['GLM_esti'] += "---------------------------------------------------------------------------------------------------------\n"
    for i in range(GLMMod.nVars):
        dic_sum['GLM_esti'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (GLMMod.x_name[i], GLMMod.Betas[i], GLMMod.std_err[i] ,GLMMod.t_stat[i][0], GLMMod.t_stat[i][1])
    
    dic_sum['GLM_esti'] += "\n"
    
    sumStr = '' 
    sumStr += dic_sum['Caption']
    sumStr += dic_sum['BeginT']
    sumStr += dic_sum['DataSource']
    sumStr += dic_sum['ModSettings']    
    sumStr += dic_sum['ModOptions']
    sumStr += dic_sum['VarSettings']    
    sumStr += dic_sum['GLMResult']
    sumStr += dic_sum['GLM_diag']    
    sumStr += dic_sum['GLM_esti']
    sumStr += dic_sum['EndT']
        
    
    GLMMod.summary = dic_sum#sumStr
    
def GWGLM(GWRMod):
    """
    get diagnostics for GWGLM model
    """
    if GWRMod.mType == 0:
        GWRMod.aic = Diagnostics.get_AIC_GWR(GWRMod)
        GWRMod.aicc = Diagnostics.get_AICc_GWR(GWRMod)
        GWRMod.bic = Diagnostics.get_BIC_GWR(GWRMod) 
        GWRMod.cv = Diagnostics.get_CV_GWR(GWRMod)
        GWRMod.R2 = Diagnostics.r2_GWR(GWRMod)
        GWRMod.R2_adj = Diagnostics.ar2_GWR(GWRMod)
    else:
        GWRMod.aic = Diagnostics.get_AIC_GWGLM(GWRMod)
        GWRMod.aicc = Diagnostics.get_AICc_GWGLM(GWRMod) 
        GWRMod.bic = Diagnostics.get_BIC_GWGLM(GWRMod) 
        #GWGLMMod.dev_res = Diagnostics.dev_res_GWGLM(GWGLMMod)
        GWRMod.dev_null = Diagnostics.dev_mod_GLM(GWRMod)
        GWRMod.pdev = 1.0 - GWRMod.dev_res/GWRMod.dev_null    
    

    summary_GWGLM(GWRMod)
    
def summary_GWGLM(GWRMod):
    """
    output summary in string
    """
    dic_sum = {}
    
    dic_sum['Caption'] = "%s\n" % ('Summary: Geographically Weighted Regression')
    dic_sum['BeginT'] = "%-21s: %s %s\n" % ('Program started at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    dic_sum['DataSource'] = "%s %s\n" % ('Data filename:', GWRMod.fle_name)
    dic_sum['ModSettings'] = "%s\n" % ('Model settings:')
    dic_sum['ModOptions'] = "%s\n" % ('Modelling options:')
    dic_sum['VarSettings'] = "%s\n" % ('Variable settings:')
    dic_sum['GlobResult'] = '' 
    dic_sum['Glob_diag'] = ''
    dic_sum['Glob_esti'] = ''  
    dic_sum['GWRResult'] = "%s\n" %('GWR (Geographically weighted regression) result')
    dic_sum['GWR_band'] = "%s\n" %('GWR (Geographically weighted regression) bandwidth selection')
    dic_sum['GWR_diag'] = "%s\n" % ('Diagnostic information') 
    dic_sum['GWR_esti_glob'] = ''
    dic_sum['GWR_esti'] = "%s\n" % ('<< Geographically varying (Local) coefficients >>')
    if GWRMod.mType == 0:
        dic_sum['GWR_anova'] = "%s\n" %('GWR ANOVA Table') 
    else:
        dic_sum['GWR_anova'] = "%s\n" %('GWR Analysis of Deviance Table') 
    dic_sum['VaryTest'] = ''
    dic_sum['l2g'] = ''
    dic_sum['g2l'] = ''
    dic_sum['newMod'] = ''
    dic_sum['EndT'] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    
    dic_sum['Caption'] += '-' * 75 + '\n'
    dic_sum['Caption'] += "\n"
    
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of observations:', GWRMod.nObs) 
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of Variables:', GWRMod.nVars) 
    dic_sum['DataSource'] += "\n"   
    
    dic_sum['ModSettings'] += '-' * 75 + '\n'
    dic_sum['ModSettings'] += "%-45s %s\n" % ('Model type:', GWRMod.mName)
    dic_sum['ModSettings'] += "%-45s %s\n" % ('Geographic kernel:', GWRMod.kernel.wName)
    #dic_sum['ModSettings'] += "\n"
    
    dic_sum['ModOptions'] += '-' * 75 + '\n'
    #dic_sum['ModOptions'] += "\n"  
       
    dic_sum['VarSettings'] += '-' * 75 + '\n'
    dic_sum['VarSettings'] += "%-60s %s\n" % ('Dependent variable:', GWRMod.y_name)
    if GWRMod.mType == 1:
            dic_sum['VarSettings'] += "%-45s %12s\n" % ('Offset variable:', GWRMod.y_off_name)    
    for xVar in GWRMod.x_name:
        dic_sum['VarSettings'] += "%-60s %s\n" % ('Independent variable with varying (Local) coefficient:', xVar)
    dic_sum['VarSettings'] += "\n"   
    
    if hasattr(GWRMod, 'GLM'):        
        dic_sum['GlobResult'] = "%s\n"  % ('Global regression result') 
        dic_sum['Glob_diag'] = ''
        dic_sum['Glob_esti'] = '' 
        dic_sum['Glob_diag'] += '-' * 75 + '\n'
        if GWRMod.mType == 0:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', GWRMod.GLM.res2)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('ML based global sigma estimate:', np.sqrt(GWRMod.GLM.sig2ML))
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Unbiased global sigma estimate:', np.sqrt(GWRMod.GLM.sigma2_nk))
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', GWRMod.GLM.dev_res)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GWRMod.GLM.aic)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('AICc:', GWRMod.GLM.aicc)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('BIC/MDL:', GWRMod.GLM.bic)
        if GWRMod.mType == 0:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('CV:', GWRMod.GLM.cv)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('R square:', GWRMod.GLM.r2)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GWRMod.GLM.ar2) 
        else:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Null deviance:', GWRMod.GLM.dev_null)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Residual deviance:', GWRMod.GLM.dev_res)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Percent deviance explained:', GWRMod.GLM.pdev)
        dic_sum['Glob_diag'] += "\n"    
    
        if GWRMod.mType == 0:
            dic_sum['Glob_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'t(Est/SE)', 'p-value')
        else:
            dic_sum['Glob_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'z(Est/SE)', 'p-value')
        dic_sum['Glob_esti'] += "---------------------------------------------------------------------------------------------------------\n"
        for i in range(GWRMod.nVars):
            dic_sum['Glob_esti'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (GWRMod.GLM.x_name[i], GWRMod.GLM.Betas[i], GWRMod.GLM.std_err[i] ,GWRMod.GLM.t_stat[i][0], GWRMod.GLM.t_stat[i][1])
        dic_sum['Glob_esti'] += "\n"   
            
    
    dic_sum['GWRResult'] += '-' * 75 + '\n'
    dic_sum['GWRResult'] += "%s\n" %('Geographic ranges')
    dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('Coordinate', 'Min', 'Max', 'Range')
    dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('-'*20, '-'*20, '-'*20, '-'*20)
    arr_coords = np.array(GWRMod.kernel.coords.values())
    arr_X = arr_coords[:,0]
    arr_Y = arr_coords[:,1]
    min_x = min(arr_X)
    max_x = max(arr_X)
    min_y = min(arr_Y)
    max_y = max(arr_Y)
    dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('X-coord', min_x, max_x, max_x-min_x)
    dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('Y-coord', min_y, max_y, max_y-min_y)
    dic_sum['GWRResult'] += "\n"
    
    dic_sum['GWR_band'] += '-' * 75 + '\n'
    dic_sum['GWR_band'] += "%-37s %20.6f\n" % ('Bandwidth size:', GWRMod.kernel.band)
    dic_sum['GWR_band'] += "\n"
    
    dic_sum['GWR_diag'] += '-' * 75 + '\n'
    if GWRMod.mType == 0:
        df_gwr = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', np.sum(GWRMod.res**2))
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (model: trace(S)):', GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (variance: trace(S' + "'" + 'S))', GWRMod.tr_STS)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (model: n - trace(S)):', GWRMod.nObs-GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (residual: n - 2trace(S) + trace(S' + "'" + 'S)):', df_gwr)#GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('ML based sigma estimate:', np.sqrt(GWRMod.sigma2_ML))
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Unbiased sigma estimate:', np.sqrt(GWRMod.sigma2_v1v2))
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', -2*GWRMod.logll)
    else:
        df_gwr = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_SWSTW
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (model: trace(S)):', GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (variance: trace(S' + "'" + 'WSW^-1))', GWRMod.tr_SWSTW)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (model: n - trace(S)):', GWRMod.nObs-GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (residual: n - 2trace(S) + trace(S' + "'" + 'WSW^-1)):', df_gwr) #tr_SWSTW S'WSW^-1
            
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GWRMod.aic)
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('AICc:', GWRMod.aicc)
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('BIC:', GWRMod.bic) 
    if GWRMod.mType == 0:
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('CV:', GWRMod.cv)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('R square:', GWRMod.R2)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GWRMod.R2_adj)
    else:
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Null deviance:', GWRMod.dev_null)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Residual deviance:', GWRMod.dev_res)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Percent deviance explained:', GWRMod.pdev)    
    dic_sum['GWR_diag'] += "\n"
    
    dic_sum['GWR_esti'] += "%s\n\n" % ('Summary statistics for varying (Local) coefficients')
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Mean' ,'STD')
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    for i in range(GWRMod.nVars):
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name[i], np.mean(GWRMod.Betas[:,i]) ,np.std(GWRMod.Betas[:,i]))
    dic_sum['GWR_esti'] += "\n"
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Min' ,'Max', 'Range')
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    for i in range(GWRMod.nVars):
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name[i], np.min(GWRMod.Betas[:,i]) ,np.max(GWRMod.Betas[:,i]), np.max(GWRMod.Betas[:,i])-np.min(GWRMod.Betas[:,i]))
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Lwr Quartile' ,'Median', 'Upr Quartile')              
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    for i in range(GWRMod.nVars):
        quan = mquantiles(GWRMod.Betas[:,i])
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name[i], quan[0],np.median(GWRMod.Betas[:,i]), quan[2])    
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Interquartile R' ,'Robust STD')              
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    for i in range(GWRMod.nVars):
        quan = mquantiles(GWRMod.Betas[:,i])
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name[i], quan[2]-quan[0], (quan[2]-quan[0])/1.349)      
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%s\n" % ('(Note: Robust STD is given by (interquartile range / 1.349) )')
    dic_sum['GWR_esti'] += "\n"
    
    dic_sum['GWR_anova'] += '-' * 75 + '\n'
    
    df_glm = GWRMod.nObs-GWRMod.nVars    
    if hasattr(GWRMod, 'GLM'):        
        if GWRMod.mType == 0:
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('Source', 'SS', 'DF', 'MS', 'F')
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('-'*20, '-'*20, '-'*20, '-'*20, '-'*20)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f\n" % ('Global Residuals', GWRMod.GLM.res2, df_glm)
            ms_imp = (GWRMod.GLM.res2-GWRMod.res2)/(df_glm-df_gwr)
            ms_gwr = GWRMod.res2/df_gwr
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR Improvement', GWRMod.GLM.res2-GWRMod.res2, df_glm-df_gwr, ms_imp)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % ('GWR Residuals', GWRMod.res2, df_gwr, ms_gwr, ms_imp/ms_gwr)
        else:
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s \n" % ('Source', 'Deviance', 'DF', 'Deviance/DF')
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s \n" % ('-'*20, '-'*20, '-'*20, '-'*20)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('Global model', GWRMod.GLM.dev_res, df_glm, GWRMod.GLM.dev_res/df_glm)
            #ms_imp = (GWRMod.OLS.res2-GWRMod.res2)/(df_gwr-df_ols)
            #ms_gwr = GWRMod.res2/df_gwr
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR model', GWRMod.dev_res, df_gwr, GWRMod.dev_res/df_gwr)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('Difference', GWRMod.GLM.dev_res-GWRMod.dev_res, df_glm-df_gwr, (GWRMod.GLM.dev_res-GWRMod.dev_res)/(df_glm-df_gwr))
    dic_sum['GWR_anova'] += "\n"    
    
    
    sumStr = '' 
    sumStr += dic_sum['Caption']
    sumStr += dic_sum['BeginT']
    sumStr += dic_sum['DataSource']
    sumStr += dic_sum['ModSettings']    
    sumStr += dic_sum['ModOptions']
    sumStr += dic_sum['VarSettings']    
    sumStr += dic_sum['GlobResult']
    sumStr += dic_sum['Glob_diag']    
    sumStr += dic_sum['Glob_esti']
    sumStr += dic_sum['GWRResult']     
    sumStr += dic_sum['GWR_band']
    sumStr += dic_sum['GWR_diag']    
    sumStr += dic_sum['GWR_esti']
    sumStr += dic_sum['GWR_anova']    
    sumStr += dic_sum['EndT']        
    
    GWRMod.summary = dic_sum#sumStr      
    
def semiGWR(semiGWRMod):
    """
    get diagnostics for semiGWR model
    """
    
     
    if semiGWRMod.mType == 0: # Gaussian model
        semiGWRMod.aic = Diagnostics.get_AIC_GWR(semiGWRMod)
        semiGWRMod.aicc = Diagnostics.get_AICc_GWR(semiGWRMod) 
        semiGWRMod.bic = Diagnostics.get_BIC_GWR(semiGWRMod) 
        semiGWRMod.cv = Diagnostics.get_CV_GWR(semiGWRMod)
        semiGWRMod.R2 = Diagnostics.r2_GWR(semiGWRMod)
        semiGWRMod.R2_adj = Diagnostics.ar2_GWR(semiGWRMod)
    else:
        semiGWRMod.aic = Diagnostics.get_AIC_GWGLM(semiGWRMod)
        semiGWRMod.aicc = Diagnostics.get_AICc_GWGLM(semiGWRMod) 
        semiGWRMod.bic = Diagnostics.get_BIC_GWGLM(semiGWRMod) 
        semiGWRMod.dev_null = Diagnostics.dev_mod_GLM(semiGWRMod)
        semiGWRMod.pdev = 1.0 - semiGWRMod.dev_res/semiGWRMod.dev_null 

    summary_semiGWR(semiGWRMod)
    
def summary_semiGWR(GWRMod):
    """
    output summary in string
    """
    dic_sum = {}
    
    dic_sum['Caption'] = "%s\n" % ('Summary: Geographically Weighted Regression')
    dic_sum['BeginT'] = "%-21s: %s %s\n" % ('Program started at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    dic_sum['DataSource'] = "%s %s\n" % ('Data filename:', GWRMod.fle_name)
    dic_sum['ModSettings'] = "%s\n" % ('Model settings:')
    dic_sum['ModOptions'] = "%s\n" % ('Modelling options:')
    dic_sum['VarSettings'] = "%s\n" % ('Variable settings:')
    dic_sum['GlobResult'] = '' 
    dic_sum['Glob_diag'] = ''
    dic_sum['Glob_esti'] = ''  
    dic_sum['GWRResult'] = "%s\n" %('GWR (Geographically weighted regression) result')
    dic_sum['GWR_band'] = "%s\n" %('GWR (Geographically weighted regression) bandwidth selection')
    dic_sum['GWR_diag'] = "%s\n" % ('Diagnostic information') 
    dic_sum['GWR_esti_glob'] = "%s\n" % ('<< Fixed (Global) coefficients >>') 
    dic_sum['GWR_esti'] = "%s\n" % ('<< Geographically varying (Local) coefficients >>')
    if GWRMod.mType == 0:
        dic_sum['GWR_anova'] = "%s\n" %('GWR ANOVA Table') 
    else:
        dic_sum['GWR_anova'] = "%s\n" %('GWR Analysis of Deviance Table') 
    dic_sum['VaryTest'] = ''
    dic_sum['l2g'] = ''
    dic_sum['g2l'] = ''
    dic_sum['newMod'] = ''
    dic_sum['EndT'] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(datetime.now()), datetime.strftime(datetime.now(),"%H:%M:%S"))
    
    dic_sum['Caption'] += '-' * 75 + '\n'
    dic_sum['Caption'] += "\n"
    
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of observations:', GWRMod.nObs) 
    dic_sum['DataSource'] += "%-45s %d\n" % ('Number of Variables:', GWRMod.nVars) 
    dic_sum['DataSource'] += "\n"   
    
    dic_sum['ModSettings'] += '-' * 75 + '\n'
    dic_sum['ModSettings'] += "%-45s %s\n" % ('Model type:', GWRMod.mName)
    dic_sum['ModSettings'] += "%-45s %s\n" % ('Geographic kernel:', GWRMod.kernel.wName)
    #dic_sum['ModSettings'] += "\n"
    
    dic_sum['ModOptions'] += '-' * 75 + '\n'
    #dic_sum['ModOptions'] += "\n"  
       
    dic_sum['VarSettings'] += '-' * 75 + '\n'
    dic_sum['VarSettings'] += "%-60s %s\n" % ('Dependent variable:', GWRMod.y_name)
    if GWRMod.mType == 1:
            dic_sum['VarSettings'] += "%-45s %12s\n" % ('Offset variable:', GWRMod.y_off_name)    
    for xVar in GWRMod.x_name_glob:
        dic_sum['VarSettings'] += "%-60s %s\n" % ('Independent variable with fixed (Global) coefficient:', xVar)
    for xVar in GWRMod.x_name_loc:
        dic_sum['VarSettings'] += "%-60s %s\n" % ('Independent variable with varying (Local) coefficient:', xVar)
    dic_sum['VarSettings'] += "\n"   
    
    if hasattr(GWRMod, 'GLM'):        
        dic_sum['GlobResult'] = "%s\n"  % ('Global regression result') 
        dic_sum['Glob_diag'] = ''
        dic_sum['Glob_esti'] = '' 
        dic_sum['Glob_diag'] += '-' * 75 + '\n'
        if GWRMod.mType == 0:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', GWRMod.GLM.res2)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('ML based global sigma estimate:', np.sqrt(GWRMod.GLM.sig2ML))
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Unbiased global sigma estimate:', np.sqrt(GWRMod.GLM.sigma2_nk))
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', GWRMod.GLM.dev_res)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GWRMod.GLM.aic)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('AICc:', GWRMod.GLM.aicc)
        dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('BIC/MDL:', GWRMod.GLM.bic)
        if GWRMod.mType == 0:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('CV:', GWRMod.GLM.cv)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('R square:', GWRMod.GLM.r2)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GWRMod.GLM.ar2) 
        else:
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Null deviance:', GWRMod.GLM.dev_null)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Residual deviance:', GWRMod.GLM.dev_res)
            dic_sum['Glob_diag'] += "%-45s %12.6f\n" % ('Percent deviance explained:', GWRMod.GLM.pdev)
        dic_sum['Glob_diag'] += "\n"    
    
        if GWRMod.mType == 0:
            dic_sum['Glob_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'t(Est/SE)', 'p-value')
        else:
            dic_sum['Glob_esti'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'z(Est/SE)', 'p-value')
        dic_sum['Glob_esti'] += "---------------------------------------------------------------------------------------------------------\n"
        for i in range(GWRMod.nVars):
            dic_sum['Glob_esti'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (GWRMod.GLM.x_name[i], GWRMod.GLM.Betas[i], GWRMod.GLM.std_err[i] ,GWRMod.GLM.t_stat[i][0], GWRMod.GLM.t_stat[i][1])
        dic_sum['Glob_esti'] += "\n" 
    
    
    dic_sum['GWRResult'] += '-' * 75 + '\n'
    dic_sum['GWRResult'] += "%s\n" %('Geographic ranges')
    dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('Coordinate', 'Min', 'Max', 'Range')
    dic_sum['GWRResult'] += "%-20s %20s %20s %20s\n" %('-'*20, '-'*20, '-'*20, '-'*20)
    arr_coords = np.array(GWRMod.kernel.coords.values())
    arr_X = arr_coords[:,0]
    arr_Y = arr_coords[:,1]
    min_x = min(arr_X)
    max_x = max(arr_X)
    min_y = min(arr_Y)
    max_y = max(arr_Y)
    dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('X-coord', min_x, max_x, max_x-min_x)
    dic_sum['GWRResult'] += "%-20s %20.6f %20.6f %20.6f\n" %('Y-coord', min_y, max_y, max_y-min_y)
    dic_sum['GWRResult'] += "\n"
    
    dic_sum['GWR_band'] += '-' * 75 + '\n'
    dic_sum['GWR_band'] += "%-37s %20.6f\n" % ('Bandwidth size:', GWRMod.kernel.band)
    dic_sum['GWR_band'] += "\n"
    
    dic_sum['GWR_diag'] += '-' * 75 + '\n'
    if GWRMod.mType == 0:
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Residual sum of squares:', GWRMod.res2)
        df_gwr = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_STS
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (model: trace(S)):', GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (variance: trace(S' + "'" + 'S))', GWRMod.tr_STS)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (model: n - trace(S)):', GWRMod.nObs-GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (residual: n - 2trace(S) + trace(S' + "'" + 'S)):', df_gwr)    
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('ML based sigma estimate:', np.sqrt(GWRMod.sigma2_ML))
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Unbiased sigma estimate:', np.sqrt(GWRMod.sigma2_v1v2))
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('-2Log-likelihood:', -2.0*GWRMod.logll)
    else:
        if GWRMod.tr_S <= GWRMod.tr_SWSTW:
            df_gwr = GWRMod.nObs-GWRMod.tr_S
            str_df = 'Degree of freedom (residual: n - trace(S)):'
            dic_sum['GWR_diag'] += "%-60s\n" % ('(Warning: trace(S) is smaller than trace(S' + 'S). It means the variance of the predictions is inadequately inflated.)')
            dic_sum['GWR_diag'] += "%-60s\n" % ('Note: n - trace(S) is used for computing the error variance as the degree of freedom.')
        else:
            df_gwr = GWRMod.nObs-2.0*GWRMod.tr_S+GWRMod.tr_SWSTW
            str_df = 'Degree of freedom (residual: n - 2trace(S) + trace(S' + "'" + 'WSW^-1)):'
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (model: trace(S)):', GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Effective number of parameters (variance: trace(S' + "'" + 'WSW^-1))', GWRMod.tr_SWSTW)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % ('Degree of freedom (model: n - trace(S)):', GWRMod.nObs-GWRMod.tr_S)
        dic_sum['GWR_diag'] += "%-60s %12.6f\n" % (str_df, df_gwr)          
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Classic AIC:', GWRMod.aic)
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('AICc:', GWRMod.aicc)
    dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('BIC:', GWRMod.bic) 
    if GWRMod.mType == 0:
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('CV:', GWRMod.cv)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('R square:', GWRMod.R2)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Adjusted R square:', GWRMod.R2_adj)
    else:
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Null deviance:', GWRMod.dev_null)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Residual deviance:', GWRMod.dev_res)
        dic_sum['GWR_diag'] += "%-45s %12.6f\n" % ('Percent deviance explained:', GWRMod.pdev)    
    dic_sum['GWR_diag'] += "\n"
    
    if GWRMod.mType == 0:
        tstat_name = 't(Est/SE)'
    else:
        tstat_name = 'z(Est/SE)'
    dic_sum['GWR_esti_glob'] += "%-20s %20s %20s %20s %20s\n" % ('Variable', 'Estimate', 'Standard Error' ,'t(Est/SE)', 'p-value')
    dic_sum['GWR_esti_glob'] += "---------------------------------------------------------------------------------------------------------\n"
    if GWRMod.mType == 0:
        n = GWRMod.nObs
        k = GWRMod.nVars_glob
        for i in range(GWRMod.nVars_glob):        
            dic_sum['GWR_esti_glob'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (GWRMod.x_name_glob[i], GWRMod.Betas_glob[i], GWRMod.std_err_glob[i],GWRMod.t_stat_glob[i], stats.t.sf(abs(GWRMod.t_stat_glob[i]),n-k)*2)
    else:
        for i in range(GWRMod.nVars_glob):  
            dic_sum['GWR_esti_glob'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % (GWRMod.x_name_glob[i], GWRMod.Betas_glob[i], GWRMod.std_err_glob[i],GWRMod.t_stat_glob[i], stats.norm.sf(abs(GWRMod.t_stat_glob[i]))*2)
    dic_sum['GWR_esti_glob'] += "\n"
    
    dic_sum['GWR_esti'] += "%s\n\n" % ('Summary statistics for varying (Local) coefficients')
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Mean' ,'STD')
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    for i in range(GWRMod.nVars_loc):
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name_loc[i], np.mean(GWRMod.Betas_loc[:,i]) ,np.std(GWRMod.Betas_loc[:,i]))
    dic_sum['GWR_esti'] += "\n"
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Min' ,'Max', 'Range')
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    for i in range(GWRMod.nVars_loc):
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name_loc[i], np.min(GWRMod.Betas_loc[:,i]) ,np.max(GWRMod.Betas_loc[:,i]), np.max(GWRMod.Betas_loc[:,i])-np.min(GWRMod.Betas_loc[:,i]))
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('Variable', 'Lwr Quartile' ,'Median', 'Upr Quartile')              
    dic_sum['GWR_esti'] += "%-20s %20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20, '-'*20)
    for i in range(GWRMod.nVars_loc):
        quan = mquantiles(GWRMod.Betas_loc[:,i])
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f %20.6f\n" % (GWRMod.x_name_loc[i], quan[0],np.median(GWRMod.Betas_loc[:,i]), quan[2])    
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('Variable', 'Interquartile R' ,'Robust STD')              
    dic_sum['GWR_esti'] += "%-20s %20s %20s\n" % ('-'*20, '-'*20 ,'-'*20)
    for i in range(GWRMod.nVars_loc):
        quan = mquantiles(GWRMod.Betas_loc[:,i])
        dic_sum['GWR_esti'] += "%-20s %20.6f %20.6f\n" % (GWRMod.x_name_loc[i], quan[2]-quan[0], (quan[2]-quan[0])/1.349)      
    dic_sum['GWR_esti'] += "\n"    
    dic_sum['GWR_esti'] += "%s\n" % ('(Note: Robust STD is given by (interquartile range / 1.349) )')
    dic_sum['GWR_esti'] += "\n"
    
    dic_sum['GWR_anova'] += '-' * 75 + '\n'    
    df_glm = GWRMod.nObs-GWRMod.nVars   
    #if hasattr(GWRMod, 'OLS'):        
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f\n" % ('Global Residuals', GWRMod.OLS.res2, df_glm)
        #ms_imp = (GWRMod.OLS.res2-GWRMod.res2)/(df_gwr-df_glm)
        #ms_gwr = GWRMod.res2/df_gwr
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR Improvement', GWRMod.OLS.res2-GWRMod.res2, df_glm-df_gwr, ms_imp)
        #dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % ('GWR Residuals', GWRMod.res2, df_gwr, ms_gwr, ms_imp/ms_gwr)
    if hasattr(GWRMod, 'GLM'):     
        if GWRMod.mType == 0: # Gaussian model
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('Source', 'SS', 'DF', 'MS', 'F')
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s %20s\n" % ('-'*20, '-'*20, '-'*20, '-'*20, '-'*20)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f\n" % ('Global Residuals', GWRMod.GLM.res2, df_glm)
            ms_imp = (GWRMod.GLM.res2-GWRMod.res2)/(df_glm-df_gwr)
            ms_gwr = GWRMod.res2/df_gwr
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR Improvement', GWRMod.GLM.res2-GWRMod.res2, df_glm-df_gwr, ms_imp)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f %20.6f\n" % ('GWR Residuals', GWRMod.res2, df_gwr, ms_gwr, ms_imp/ms_gwr)
        else:
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s \n" % ('Source', 'Deviance', 'DF', 'Deviance/DF')
            dic_sum['GWR_anova'] += "%-20s %20s %20s %20s \n" % ('-'*20, '-'*20, '-'*20, '-'*20)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('Global model', GWRMod.GLM.dev_res, df_glm, GWRMod.GLM.dev_res/df_glm)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('GWR model', GWRMod.dev_res, df_gwr, GWRMod.dev_res/df_gwr)
            dic_sum['GWR_anova'] += "%-20s %20.6f %20.6f %20.6f\n" % ('Difference', GWRMod.GLM.dev_res-GWRMod.dev_res, df_glm-df_gwr, (GWRMod.GLM.dev_res-GWRMod.dev_res)/(df_glm-df_gwr))
    dic_sum['GWR_anova'] += "\n"
    
    sumStr = '' 
    sumStr += dic_sum['Caption']
    sumStr += dic_sum['BeginT']
    sumStr += dic_sum['DataSource']
    sumStr += dic_sum['ModSettings']    
    sumStr += dic_sum['ModOptions']
    sumStr += dic_sum['VarSettings']    
    sumStr += dic_sum['GlobResult']
    sumStr += dic_sum['Glob_diag'] 
    sumStr += dic_sum['Glob_esti']
    sumStr += dic_sum['GWRResult']     
    sumStr += dic_sum['GWR_band']
    sumStr += dic_sum['GWR_diag']  
    sumStr += dic_sum['GWR_esti_glob']
    sumStr += dic_sum['GWR_esti']
    sumStr += dic_sum['GWR_anova']    
    sumStr += dic_sum['EndT']
        
    
    GWRMod.summary = dic_sum