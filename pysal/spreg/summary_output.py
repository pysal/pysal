"""
Internal helper files for user output.
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, Jing Yao jingyao@asu.edu"

import textwrap as TW
import numpy as np
import copy as COPY
import diagnostics as diagnostics
import diagnostics_tsls as diagnostics_tsls
import diagnostics_sp as diagnostics_sp
import pysal
import scipy
from scipy.sparse.csr import csr_matrix

__all__ = []


###############################################################################
############### Primary functions for running summary diagnostics #############
###############################################################################

"""
This section contains one function for each user level regression class. These
are called directly from the user class. Each one mixes and matches smaller
functions located later in this module.
"""

def OLS(reg, vm, w, nonspat_diag, spat_diag, moran, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_ols(reg, reg.robust)
    if nonspat_diag:
        # compute diagnostics
        reg.sig2ML = reg.sig2n  
        reg.f_stat = diagnostics.f_stat(reg)  
        reg.logll = diagnostics.log_likelihood(reg) 
        reg.aic = diagnostics.akaike(reg) 
        reg.schwarz = diagnostics.schwarz(reg) 
        reg.mulColli = diagnostics.condition_index(reg)
        reg.jarque_bera = diagnostics.jarque_bera(reg)
        reg.breusch_pagan = diagnostics.breusch_pagan(reg)
        reg.koenker_bassett = diagnostics.koenker_bassett(reg)
        reg.white = diagnostics.white(reg)
        # organize summary output
        reg.__summary['summary_nonspat_diag_1'] = summary_nonspat_diag_1(reg)
        reg.__summary['summary_nonspat_diag_2'] = summary_nonspat_diag_2(reg)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_ols(reg, w, moran)
    if regimes:
        summary_regimes(reg)
    summary(reg=reg, vm=vm, instruments=False, nonspat_diag=nonspat_diag, spat_diag=spat_diag)

def OLS_multi(reg, multireg, vm, nonspat_diag, spat_diag, moran, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_ols(mreg, mreg.robust)
        if nonspat_diag:
            # compute diagnostics
            mreg.sig2ML = mreg.sig2n  
            mreg.f_stat = diagnostics.f_stat(mreg)  
            mreg.logll = diagnostics.log_likelihood(mreg) 
            mreg.aic = diagnostics.akaike(mreg) 
            mreg.schwarz = diagnostics.schwarz(mreg) 
            mreg.mulColli = diagnostics.condition_index(mreg)
            mreg.jarque_bera = diagnostics.jarque_bera(mreg)
            mreg.breusch_pagan = diagnostics.breusch_pagan(mreg)
            mreg.koenker_bassett = diagnostics.koenker_bassett(mreg)
            mreg.white = diagnostics.white(mreg)
            # organize summary output
            mreg.__summary['summary_nonspat_diag_1'] = summary_nonspat_diag_1(mreg)
            mreg.__summary['summary_nonspat_diag_2'] = summary_nonspat_diag_2(mreg)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_ols(mreg, mreg.w, moran)
        if regimes:
            summary_regimes(mreg,chow=False)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=False, nonspat_diag=nonspat_diag, spat_diag=spat_diag)

def TSLS(reg, vm, w, spat_diag, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, reg.robust)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_instruments(reg, w)
    # build coefficients table body
    build_coefs_body_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=spat_diag)

def TSLS_multi(reg, multireg, vm, spat_diag, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, mreg.robust)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_instruments(mreg, mreg.w)
        # build coefficients table body
        build_coefs_body_instruments(mreg)
        if regimes:
            summary_regimes(mreg,chow=False)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=spat_diag)

def GM_Lag(reg, vm, w, spat_diag, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, reg.robust)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_instruments(reg, w)
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=spat_diag)

def GM_Lag_multi(reg, multireg, vm, spat_diag, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, mreg.robust)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_instruments(mreg, mreg.w)
        # build coefficients table body
        summary_coefs_yend(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg,chow=False)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=spat_diag)

def GM_Error(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Error_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg,chow=False)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        summary_coefs_yend(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg,chow=False)
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Error_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Error_Hom_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg,chow=False)
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error_Hom_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        summary_coefs_yend(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg,chow=False)
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Error_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, 'het')
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Error_Het_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, 'het')
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg,chow=False)
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=False, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, 'het')
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Endog_Error_Het_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, 'het')
        # build coefficients table body
        summary_coefs_yend(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg,chow=False)
    reg.__summary = {}
    summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Combo(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, None)
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    summary_warning(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Combo_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, None)
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def GM_Combo_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, 'het')
    # build coefficients table body
    summary_coefs_yend(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True, nonspat_diag=False, spat_diag=False)

def Probit(reg, vm, w, spat_diag):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None) 
    # organize summary output
    if spat_diag:
        reg.__summary['summary_spat_diag'] = summary_spat_diag_probit(reg)
    reg.__summary['summary_r2'] = "%-21s: %3.2f\n" % ('% correctly predicted',reg.predpc)
    reg.__summary['summary_r2'] += "%-21s: %3.4f\n" % ('Log-Likelihood',reg.logl)    
    reg.__summary['summary_r2'] += "%-21s: %3.4f\n" % ('LR test',reg.LR[0])
    reg.__summary['summary_r2'] += "%-21s: %3.4f\n" % ('LR test (p-value)',reg.LR[1])
    if reg.warning:
         reg.__summary['summary_r2'] += "\nMaximum number of iterations exceeded or gradient and/or function calls not changing\n"
    # build coefficients table body
    beta_position = summary_coefs_allx(reg, reg.z_stat)
    reg.__summary['summary_other_mid']= summary_coefs_slopes(reg)
    summary(reg=reg, vm=vm, instruments=False, short_intro=True, spat_diag=spat_diag)

##############################################################################




##############################################################################
############### Helper functions for running summary diagnostics #############
##############################################################################

def beta_diag_ols(reg, robust):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.t_stat = diagnostics.t_stat(reg)
    reg.r2 = diagnostics.r2(reg)    
    reg.ar2 = diagnostics.ar2(reg)   
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 't'
    reg.__summary['summary_r2'] = "%-20s:%12.6f\n%-20s:%12.4f\n" % ('R-squared',reg.r2,'Adjusted R-squared',reg.ar2)
    # build coefficients table body
    position = summary_coefs_allx(reg, reg.t_stat)

def beta_diag(reg, robust):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.z_stat = diagnostics.t_stat(reg, z_stat=True)
    reg.pr2 = diagnostics_tsls.pr2_aspatial(reg)
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 'z'
    reg.__summary['summary_r2'] = "%-20s:%12.6f\n" % ('Pseudo R-squared',reg.pr2)

def beta_diag_lag(reg, robust):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.z_stat = diagnostics.t_stat(reg, z_stat=True)
    reg.pr2 = diagnostics_tsls.pr2_aspatial(reg)
    reg.pr2_e = diagnostics_tsls.pr2_spatial(reg)
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 'z'
    reg.__summary['summary_r2'] = "%-20s:      %5.4f\n" % ('Pseudo R-squared',reg.pr2)
    reg.__summary['summary_r2'] += "%-20s:  %5.4f\n" % ('Spatial Pseudo R-squared',reg.pr2_e)

def build_coefs_body_instruments(reg):
    beta_position = summary_coefs_allx(reg, reg.z_stat)
    summary_coefs_yend(reg, reg.z_stat)
    summary_coefs_instruments(reg)

def spat_diag_ols(reg, w, moran):
    # compute diagnostics
    lm_tests = diagnostics_sp.LMtests(reg, w)
    reg.lm_error = lm_tests.lme
    reg.lm_lag = lm_tests.lml
    reg.rlm_error = lm_tests.rlme
    reg.rlm_lag = lm_tests.rlml
    reg.lm_sarma = lm_tests.sarma
    if moran:
        moran_res = diagnostics_sp.MoranRes(reg, w, z=True)
        reg.moran_res = moran_res.I, moran_res.zI, moran_res.p_norm 
    # organize summary output
    reg.__summary['summary_spat_diag'] = summary_spat_diag_ols(reg, moran)

def spat_diag_instruments(reg, w):
    # compute diagnostics
    cache = diagnostics_sp.spDcache(reg, w)
    mi, ak, ak_p = diagnostics_sp.akTest(reg, w, cache)
    reg.ak_test = ak, ak_p
    # organize summary output
    reg.__summary['summary_spat_diag'] = "%-27s      %2d    %12.6f       %9.7f\n" % ("Anselin-Kelejian Test", 1, reg.ak_test[0], reg.ak_test[1])

def summary(reg, vm, instruments, short_intro=False, nonspat_diag=False, spat_diag=False, other_end=False):
    summary = summary_open()
    summary += summary_intro(reg,short_intro)
    summary += reg.__summary['summary_r2']
    if nonspat_diag:
        summary += reg.__summary['summary_nonspat_diag_1']
    summary += summary_coefs_intro(reg)
    summary += reg.__summary['summary_coefs']
    summary += "------------------------------------------------------------------------------------\n"
    if instruments:
        summary += reg.__summary['summary_coefs_instruments']
    try:
        summary += reg.__summary['summary_other_mid']
    except:
        pass
    try:
        summary += reg.__summary['summary_chow']
    except:
        pass
    if nonspat_diag:
        summary += reg.__summary['summary_nonspat_diag_2']
    if spat_diag:
        summary += summary_spat_diag_intro()
        summary += reg.__summary['summary_spat_diag']
    if vm:
        summary += summary_vm(reg, instruments)
    if other_end:
        summary += reg.__summary['summary_other_end']        
    summary += summary_close()
    reg.summary = summary

def summary_multi(reg, multireg, vm, instruments, short_intro=False, nonspat_diag=False, spat_diag=False, other_end=False):
    summary = summary_open(multi=True)
    for m in multireg:
        mreg = multireg[m]
        summary += "----------\n\n"
        summary += summary_intro(mreg,short_intro)
        summary += mreg.__summary['summary_r2']
        if nonspat_diag:
            summary += mreg.__summary['summary_nonspat_diag_1']
        summary += summary_coefs_intro(mreg)
        summary += mreg.__summary['summary_coefs']
        summary += "------------------------------------------------------------------------------------\n"
        if instruments:
            summary += mreg.__summary['summary_coefs_instruments']
        try:
            summary += mreg.__summary['summary_other_mid']
        except:
            pass
        if m == multireg.keys()[-1]:
            try:
                summary += reg.__summary['summary_other_mid']
            except:
                pass
            try:
                summary += reg.__summary['summary_chow']
            except:
                pass
        if nonspat_diag:
            summary += mreg.__summary['summary_nonspat_diag_2']
        if spat_diag:
            summary += summary_spat_diag_intro()
            summary += mreg.__summary['summary_spat_diag']
        if vm:
            summary += summary_vm(mreg, instruments)
        if other_end:
            summary += mreg.__summary['summary_other_end']
        if m == multireg.keys()[-1]:
            try:
                summary += reg.__summary['summary_other_end']
            except:
                pass
    summary += summary_close()
    reg.summary = summary

##############################################################################




##############################################################################
############### Guts of the summary printout #################################
##############################################################################

"""
This section contains the pieces needed to put together the summary printout.
"""
def summary_open(multi=False):
    strSummary = ""
    strSummary += "REGRESSION\n"
    if not multi:
        strSummary += "----------\n"
    return strSummary

def summary_intro(reg,short):
    title = "SUMMARY OF OUTPUT: " + reg.title + "\n"
    strSummary = title
    strSummary += "-" * (len(title)-1) + "\n"
    strSummary += "%-20s:%12s\n" % ('Data set',reg.name_ds)
    if reg.name_w:
        strSummary += "%-20s:%12s\n" % ('Weights matrix',reg.name_w)
    strSummary += "%-20s:%12s               %-22s:%12d\n" % ('Dependent Variable',reg.name_y,'Number of Observations',reg.n)
    if not short:
        strSummary += "%-20s:%12.4f               %-22s:%12d\n" % ('Mean dependent var',reg.mean_y,'Number of Variables',reg.k)
        strSummary += "%-20s:%12.4f               %-22s:%12d\n" % ('S.D. dependent var',reg.std_y,'Degrees of Freedom',reg.n-reg.k)
    strSummary += '\n'
    return strSummary

def summary_coefs_intro(reg):
    strSummary = "\n"
    if reg.__summary['summary_std_err']:
        if reg.__summary['summary_std_err'].lower() == 'white':
            strSummary += "White Standard Errors\n"
        elif reg.__summary['summary_std_err'].lower() == 'hac':
            strSummary += "HAC Standard Errors; Kernel Weights: " + reg.name_gwk +"\n"
        #elif reg.__summary['summary_std_err'].lower() == 'het':
            #strSummary += "Heteroskedastic Corrected Standard Errors\n"
    strSummary += "------------------------------------------------------------------------------------\n"
    strSummary += "            Variable     Coefficient       Std.Error     %1s-Statistic     Probability\n" %(reg.__summary['summary_zt'])
    strSummary += "------------------------------------------------------------------------------------\n"
    return strSummary

def summary_coefs_allx(reg, zt_stat):
    strSummary = ""
    indices = [0]+(np.argsort(reg.name_x[1:])+1).tolist()
    for i in indices:        
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                     % (reg.name_x[i],reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
    reg.__summary['summary_coefs'] = strSummary
    return i

def summary_coefs_somex(reg, zt_stat):
    """This is a special case needed for models that do not have inference on
    the lambda term
    """
    strSummary = ""
    indices = [0]+(np.argsort(reg.name_x[1:-1])+1).tolist()
    for i in indices:        
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                     % (reg.name_x[i],reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])
    reg.__summary['summary_coefs'] = strSummary
    return i


def summary_coefs_yend(reg, zt_stat, lambd=False):
    strSummary = ""
    if lambd:
        indices = [0]+(np.argsort(reg.name_z[1:-1])+1).tolist()
    else:
        indices = [0]+(np.argsort(reg.name_z[1:])+1).tolist() 
    for i in indices:
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                     % (reg.name_z[i],reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])              
    reg.__summary['summary_coefs'] = strSummary
    
def summary_coefs_lambda(reg, zt_stat):
    try:
        name_var = reg.name_z
    except:
        name_var = reg.name_x
    if len(reg.betas) == len(zt_stat):
        reg.__summary['summary_coefs'] += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                           % (name_var[-1],reg.betas[-1][0],reg.std_err[-1],zt_stat[-1][0],zt_stat[-1][1])
    else:
        reg.__summary['summary_coefs'] += "%20s    %12.7f    \n" % (name_var[-1],reg.betas[-1][0])

def summary_coefs_instruments(reg):
    """Generates a list of the instruments used.
    """
    insts = "Instruments: "
    for name in sorted(reg.name_q):
        insts += name + ", "
    text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
    insts = text_wrapper.fill(insts[:-2])
    insts += "\n"
    inst2 = "Instrumented: "
    for name in sorted(reg.name_yend):
        inst2 += name + ", "
    text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
    inst2 = text_wrapper.fill(inst2[:-2])    
    inst2 += "\n"
    inst2 += insts
    reg.__summary['summary_coefs_instruments'] = inst2

def summary_regimes(reg,chow=True):
    """Lists the regimes variable used.
    """
    try:
        reg.__summary['summary_other_mid'] += "Regimes variable: %s\n" %reg.name_regimes
    except:
        reg.__summary['summary_other_mid'] = "Regimes variable: %s\n" %reg.name_regimes
    if chow:
        summary_chow(reg)

def summary_chow(reg):
    reg.__summary['summary_chow'] = "\nREGIMES DIAGNOSTICS - CHOW TEST\n"
    reg.__summary['summary_chow'] += "                 VARIABLE        DF        VALUE           PROB\n"
    if reg.cols2regi == 'all':
        names_chow = reg.name_x_r[1:]
    else:
        names_chow = [reg.name_x_r[1:][i] for i in np.where(reg.cols2regi)[0]]
    if reg.constant_regi=='many':
        indices = [0]+(np.argsort(names_chow)+1).tolist()
        names_chow = ['CONSTANT']+names_chow
    else:
        indices = (np.argsort(names_chow)).tolist() 
    for i in indices:    
        reg.__summary['summary_chow'] += "%25s        %2d    %12.6f        %9.7f\n" %(names_chow[i],reg.nr-1,reg.chow.regi[i,0],reg.chow.regi[i,1])
    reg.__summary['summary_chow'] += "%25s        %2d    %12.6f        %9.7f\n" %('Global test',reg.kr*(reg.nr-1),reg.chow.joint[0],reg.chow.joint[1])

def summary_warning(reg):
    try:
        try:
            reg.__summary['summary_other_mid'] += reg.warning+"\n"
        except:
            reg.__summary['summary_other_mid'] = reg.warning+"\n" 
    except:
        pass

def summary_coefs_slopes(reg):
    strSummary = "\nMARGINAL EFFECTS\n"
    if reg.scalem == 'phimean':
        strSummary += "Method: Mean of individual marginal effects\n"        
    elif reg.scalem == 'xmean':
        strSummary += "Method: Marginal effects at variables mean\n"   
    strSummary += "------------------------------------------------------------------------------------\n"
    strSummary += "            Variable           Slope       Std.Error     %1s-Statistic     Probability\n" %(reg.__summary['summary_zt'])
    strSummary += "------------------------------------------------------------------------------------\n"
    indices = np.argsort(reg.name_x[1:]).tolist()
    for i in indices:        
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                     % (reg.name_x[i+1],reg.slopes[i][0],reg.slopes_std_err[i],reg.slopes_z_stat[i][0],reg.slopes_z_stat[i][1])
    return strSummary+"\n\n"

def summary_r2(reg, ols, spatial_lag):
    if ols:
        strSummary = "%-20s:%12.6f\n%-20s:%12.4f\n" % ('R-squared',reg.r2,'Adjusted R-squared',reg.ar2)
    else:
        strSummary = "%-20s:%12.6f\n" % ('Pseudo R-squared',reg.pr2)
        if spatial_lag:
            if reg.pr2_e != None: 
                strSummary += "%-20s:%12.6f\n" % ('Spatial Pseudo R-squared',reg.pr2_e)
    return strSummary

def summary_nonspat_diag_1(reg):
    strSummary = ""
    strSummary += "%-20s:%12.3f               %-22s:%12.4f\n" % ('Sum squared residual',reg.utu,'F-statistic',reg.f_stat[0])
    strSummary += "%-20s:%12.3f               %-22s:%12.4g\n" % ('Sigma-square',reg.sig2,'Prob(F-statistic)',reg.f_stat[1])
    strSummary += "%-20s:%12.3f               %-22s:%12.3f\n" % ('S.E. of regression',np.sqrt(reg.sig2),'Log likelihood',reg.logll)
    strSummary += "%-20s:%12.3f               %-22s:%12.3f\n" % ('Sigma-square ML',reg.sig2ML,'Akaike info criterion',reg.aic)
    strSummary += "%-20s:%12.4f               %-22s:%12.3f\n" % ('S.E of regression ML',np.sqrt(reg.sig2ML),'Schwarz criterion',reg.schwarz)
    return strSummary
    
def summary_nonspat_diag_2(reg):
    strSummary = ""
    strSummary += "\nREGRESSION DIAGNOSTICS\n"
    if reg.mulColli:
        strSummary += "MULTICOLLINEARITY CONDITION NUMBER %16.6f\n\n" % (reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d  %14.6f        %9.7f\n\n" % ('Jarque-Bera',reg.jarque_bera['df'],reg.jarque_bera['jb'],reg.jarque_bera['pvalue'])
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ('Breusch-Pagan test',reg.breusch_pagan['df'],reg.breusch_pagan['bp'],reg.breusch_pagan['pvalue'])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ('Koenker-Bassett test',reg.koenker_bassett['df'],reg.koenker_bassett['kb'],reg.koenker_bassett['pvalue'])
    if reg.white:
        strSummary += "\nSPECIFICATION ROBUST TEST\n"
        if len(reg.white)>3:
            strSummary += reg.white+'\n'
        else:
            strSummary += "TEST                             DF        VALUE           PROB\n"
            strSummary += "%-27s      %2d    %12.6f        %9.7f\n" %('White',reg.white['df'],reg.white['wh'],reg.white['pvalue'])
    return strSummary

def summary_spat_diag_intro():
    strSummary = ""
    strSummary += "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
    strSummary += "TEST                           MI/DF       VALUE           PROB\n" 
    return strSummary

def summary_spat_diag_ols(reg, moran):
    strSummary = ""
    if moran:
        strSummary += "%-27s  %8.4f     %8.6f        %9.7f\n" % ("Moran's I (error)", reg.moran_res[0], reg.moran_res[1], reg.moran_res[2])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ("Lagrange Multiplier (lag)", 1, reg.lm_lag[0], reg.lm_lag[1])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ("Robust LM (lag)", 1, reg.rlm_lag[0], reg.rlm_lag[1])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ("Lagrange Multiplier (error)", 1, reg.lm_error[0], reg.lm_error[1])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n" % ("Robust LM (error)", 1, reg.rlm_error[0], reg.rlm_error[1])
    strSummary += "%-27s      %2d    %12.6f        %9.7f\n\n" % ("Lagrange Multiplier (SARMA)", 2, reg.lm_sarma[0], reg.lm_sarma[1])
    return strSummary

def summary_spat_diag_probit(reg):
    strSummary = ""
    strSummary += "%-27s      %2d    %12.6f       %9.7f\n" % ("Kelejian-Prucha (error)", 1, reg.KP_error[0], reg.KP_error[1])
    strSummary += "%-27s      %2d    %12.6f       %9.7f\n" % ("Pinkse (error)", 1, reg.Pinkse_error[0], reg.Pinkse_error[1])
    strSummary += "%-27s      %2d    %12.6f       %9.7f\n\n" % ("Pinkse-Slade (error)", 1, reg.PS_error[0], reg.PS_error[1])
    return strSummary

def summary_vm(reg, instruments):
    strVM = "\n"
    strVM += "COEFFICIENTS VARIANCE MATRIX\n"
    strVM += "----------------------------\n"
    if instruments:
        for name in reg.name_z:
            strVM += "%12s" % (name)
    else:
        for name in reg.name_x:
            strVM += "%12s" % (name)
    strVM += "\n"
    nrow = reg.vm.shape[0]
    ncol = reg.vm.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            strVM += "%12.6f" % (reg.vm[i][j]) 
        strVM += "\n"
    return strVM

def summary_pred(reg):
    strPred = "\n\n"
    strPred += "%16s%16s%16s%16s\n" % ('OBS',reg.name_y,'PREDICTED','RESIDUAL')
    for i in range(reg.n):
        strPred += "%16d%16.5f%16.5f%16.5f\n" % (i+1,reg.y[i][0],reg.predy[i][0],reg.u[i][0])
    return strPred
            
def summary_close():
    return "================================ END OF REPORT ====================================="
    
##############################################################################


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



