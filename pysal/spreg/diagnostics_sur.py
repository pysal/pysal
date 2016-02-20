"""
Diagnostics for SUR and 3SLS estimation
"""

__author__= "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"
            

import numpy as np
import scipy.stats as stats
import numpy.linalg as la
from sur_utils import sur_dict2mat,sur_mat2dict,sur_corr
from regimes import buildR1var,wald_test


__all__ = ['sur_setp','sur_lrtest','sur_lmtest','lam_setp','surLMe']


def sur_setp(bigB,varb):
    ''' Utility to compute standard error, t and p-value
    
    Parameters
    ----------
    bigB    : dictionary of regression coefficient estimates,
              one vector by equation
    varb    : variance-covariance matrix of coefficients
    
    Returns
    -------
    surinfdict : dictionary with standard error, t-value, and
                 p-value array, one for each equation
    
    '''
    vvb = varb.diagonal()
    n_eq = len(bigB.keys())
    bigK = np.zeros((n_eq,1),dtype=np.int_)
    for r in range(n_eq):
        bigK[r] = bigB[r].shape[0]
    b = sur_dict2mat(bigB)
    se = np.sqrt(vvb)
    se.resize(len(se),1)
    t = np.divide(b,se)
    tp = stats.norm.sf(abs(t))*2
    surinf = np.hstack((se,t,tp))
    surinfdict = sur_mat2dict(surinf,bigK)
    return surinfdict
    
def lam_setp(lam,vm):
    """Standard errors, t-test and p-value for lambda in SUR Error ML
    
    Parameters
    ----------
    lam        : n_eq x 1 array with ML estimates for spatial error
                 autoregressive coefficient
    vm         : n_eq x n_eq subset of variance-covariance matrix for
                 lambda and Sigma in SUR Error ML
                 (needs to be subset from full vm)
                 
    Returns
    -------
               : tuple with arrays for standard error, t-value and p-value
                 (each element in the tuple is an n_eq x 1 array)
        
    """
    vvb = vm.diagonal()
    se = np.sqrt(vvb)
    se.resize(len(se),1)
    t = np.divide(lam,se)
    tp = stats.norm.sf(abs(t))*2
    return (se,t,tp)

def sur_lrtest(n,n_eq,ldetS0,ldetS1):
    ''' Likelihood Ratio test on off-diagonal elements of Sigma
    
        Parameters
        ----------
        n        : cross-sectional dimension (number of observations for an equation)
        n_eq     : number of equations
        ldetS0   : log determinant of Sigma for OLS case
        ldetS1   : log determinant of Sigma for SUR case (should be iterated)
        
        Returns
        -------
        (lrtest,M,pvalue) : tupel with value of test statistic (lrtest),
                            degrees of freedom (M, as an integer)
                            p-value
    
    '''
    M = n_eq * (n_eq - 1)/2.0
    lrtest = n * (ldetS0 - ldetS1)
    pvalue = stats.chi2.sf(lrtest,M)
    return (lrtest,int(M),pvalue)

    
def sur_lmtest(n,n_eq,sig):
    ''' Lagrange Multiplier test on off-diagonal elements of Sigma
    
        Parameters
        ----------
        n        : cross-sectional dimension (number of observations for an equation)
        n_eq     : number of equations
        sig      : inter-equation covariance matrix for null model (OLS)
        
        Returns
        -------
        (lmtest,M,pvalue) : tupel with value of test statistic (lmtest),
                            degrees of freedom (M, as an integer)
                            p-value
    
    '''
    R = sur_corr(sig)
    tr = np.trace(np.dot(R.T,R))
    M = n_eq * (n_eq - 1)/2.0
    lmtest = (n/2.0) * (tr - n_eq)
    pvalue = stats.chi2.sf(lmtest,M)
    return (lmtest,int(M),pvalue)

    
def surLMe(n_eq,WS,bigE,sig):
    """Lagrange Multiplier test on error spatial autocorrelation in SUR
    
    Parameters
    ----------
    n_eq       : number of equations
    WS         : spatial weights matrix in sparse form
    bigE       : n x n_eq matrix of residuals by equation
    sig        : cross-equation error covariance matrix
    
    Returns
    -------
    (LMe,n_eq,pvalue) : tupel with value of statistic (LMe), degrees
                        of freedom (n_eq) and p-value
    
    """
    # spatially lagged residuals
    WbigE = WS * bigE
    # score
    EWE = np.dot(bigE.T,WbigE)
    sigi = la.inv(sig)
    SEWE = sigi * EWE
    score = SEWE.sum(axis=1)
    score.resize(n_eq,1)
    # trace terms
    WW = WS * WS
    trWW = np.sum(WW.diagonal())
    WTW = WS.T * WS
    trWtW = np.sum(WTW.diagonal())
    # denominator
    SiS = sigi * sig
    Tii = trWW * np.identity(n_eq)
    tSiS = trWtW * SiS
    denom = Tii + tSiS
    idenom = la.inv(denom)
    # test statistic
    LMe = np.dot(np.dot(score.T,idenom),score)[0][0]
    pvalue = stats.chi2.sf(LMe,n_eq)
    return (LMe,n_eq,pvalue)
    
def sur_chow(n_eq,bigK,bSUR,varb):
    """test on constancy of regression coefficients across equations in
       a SUR specification
       
       Note: requires a previous check on constancy of number of coefficients
             across equations; no other checks are carried out, so it is possible
             that the results are meaningless if the variables are not listed in
             the same order in each equation.
             
       Parameters
       ----------
       n_eq       : integer, number of equations
       bigK       : array with the number of variables by equation (includes constant)
       bSUR       : dictionary with the SUR regression coefficients by equation
       varb       : array with the variance-covariance matrix for the SUR regression
                    coefficients
                    
       Returns
       -------
       test       : a list with for each coefficient (in order) a tuple with the
                    value of the test statistic, the degrees of freedom, and the
                    p-value
    
    """
    kr = bigK[0][0]
    test = []
    bb = sur_dict2mat(bSUR)
    kf = 0
    nr = n_eq
    df = n_eq - 1
    for i in range(kr):
        Ri = buildR1var(i,kr,kf,0,nr)
        tt,p = wald_test(bb,Ri,np.zeros((df,1)),varb)
        test.append((tt,df,p))
    return test
    
def sur_joinrho(n_eq,bigK,bSUR,varb):
    """Test on joint significance of spatial autoregressive coefficient in SUR
    
       Parameters
       ----------
       n_eq       : integer, number of equations
       bigK       : n_eq x 1 array with number of variables by equation
                    (includes constant term, exogenous and endogeneous and 
                    spatial lag)
       bSUR       : dictionary with regression coefficients by equation, with
                    the spatial autoregressive term as last
       varb       : variance-covariance matrix for regression coefficients
       
       Returns
       -------
                  : tuple with test statistic, degrees of freedom, p-value
        
    """
    bb = sur_dict2mat(bSUR)
    R = np.zeros((n_eq,varb.shape[0]))
    q = np.zeros((n_eq,1))
    kc = -1
    for i in range(n_eq):
        kc = kc + bigK[i]
        R[i,kc] = 1
    w,p = wald_test(bb,R,q,varb)
    return (w,n_eq,p)