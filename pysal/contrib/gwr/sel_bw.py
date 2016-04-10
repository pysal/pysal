# GWR Bandwidth selection
__author__ = "Qunshan Zhao qszhao@asu.edu"

# For Model estimation
import kernels
import numpy as np
#from M_GWGLM import GWGLM_Base
#from M_semiGWR import semiGWR_Base
#from diagnostics import get_AICc_GWR, get_AIC_GWR, get_BIC_GWR, get_CV_GWR, get_AICc_GWGLM, get_AIC_GWGLM, get_BIC_GWGLM
#import random

#get_kernel = {0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare} # define kernel function
#getDiag_GWR = {0: get_AICc_GWR,1:get_AIC_GWR, 2:get_BIC_GWR,3: get_CV_GWR} # bandwidth selection criteria
#getDiag_GWGLM = {0: get_AICc_GWGLM,1:get_AIC_GWGLM, 2:get_BIC_GWGLM}
#get_GWRModel = {0: GWR_Gaussian_Base, 1: GWR_Poisson_Base, 2: GWR_Logistic_Base} # GWR model type

def ini_band_dist(dist, nVars, wType, maxVal=0.0, minVal=0.0):
    """
    get initial bandwidth using distance
    method: from Tomoki, nn: X = 40 + 2 * (# of explanatory variables)  
                         distance: X = 50 + 2 * (# of explanatory variables)
    
    Arguments:
        dist           : dictionary
                         dist={0:{0:d_00,1:d_01,2:d_02...},1:{0:d_01,1:d_11, 2:d_12,...},...}, n*n
        nVars          : integer
                         number of Xs
        wType          : integer
                         type of kernel
        maxVal         : float
                         maximum value of bandwidth
        minVal         : float
                         minimum value of bandwidth
    
    Return:
           a: minimum bandwidth
           c: maximum bandwidth
    """
    npt = len(dist.keys())    
    
    if wType == 1 or wType == 3:# 1: bandwidth is nn
        a = 40 + 2 * nVars
        c = npt
    else: # 2: bandwidth is distance   
        listMax = []
        listMin = []
        tmpList = []
        nn = 40 + 2 * nVars
        for val in dist.values():
            tmpList = val.values()
            tmpList.sort()
            listMin.append(tmpList[nn-1]) # assuming minimum # of obs included in local calibration is 0.05*n ??
            listMax.append(tmpList[npt-1])
        a = min(listMin)/2.0 #51406.9012973939  
        c = max(listMax)/2.0 #279452.01996765
        
    # adjust a,c based on user specified values
    if a < minVal:
        a = minVal
    if c > maxVal and maxVal > 0:
        c = maxVal
                
    return a,c
    

def f_Golden(y, x_glob, x_loc, y_off, coords, mType, wType,criterion, maxVal, minVal, tol, maxIter=200,flag=0):
    """
    Golden section search
    
    Arguments
    ----------
        y              : array
                         n*1, dependent variable.
        x_glob         : array
                         n*k1, fixed independent variable.
        x_local        : array
                         n*k2, local independent variable, including constant.
        y_off          : array
                         n*1, offset variable for Poisson model
        coords         : dictionary
                         including (x,y) coordinates involved in the weight evaluation (including point i)  
        mType          : integer
                         GWR model type, 0: Gaussian, 1: Poisson, 2: Logistic
        wType          : integer
                         kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare 
        criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
        maxVal         : float
                         maximum value used in bandwidth searching
        minVal         : float
                         minimum value used in bandwidth searching
        tol            : float
                         tolerance used to determine convergence 
        maxIter        : integer
                         maximum number of iteration if convergence cannot arrive at the tolerance
        flag           : integer
                         distance type
    
    Return:
           opt_band   : float
                        optimal bandwidth
           opt_weit   : kernel
                        optimal kernel
           output     : list of tuple
                        report searching process, keep bandwidth and score, [(bandwidth, score),(bandwidth, score),...]
    """
    dist = Kernel.get_pairDist(coords,flag) #get pairwise distance between points
    
    # 1 set range of bandwidth
    if x_glob is None:
        nVar_glob = 0
    else:
        nVar_glob = len(x_glob[0])
        
    if x_loc is None:
        nVar_loc = 0
    else:
        nVar_loc = len(x_loc[0])
        
    nVars = nVar_glob + nVar_loc
    
    a,c = ini_band_dist(dist, nVars, wType, maxVal, minVal)
    
    # 2 get initial b value
    output = [] 
   
    lamda = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
    
    # get b and d
    b = a + lamda * abs(c-a) #distance or nn based on wType
    d = c - lamda * abs(c-a) # golden section
    if wType == 1 or wType == 3: # bandwidth is nn
        b = round(b,0)     
        d = round(d,0)       
            
    # 3 loop
    pre_opt = 0.0
    diff = 1.0e9  
    nIter  = 0
    while abs(diff) > tol and nIter < maxIter:           
        nIter += 1
        
        # 3.1 create kernel
        weit_a = Kernel.GWR_W(coords, a, wType, dist) 
        weit_b = Kernel.GWR_W(coords, b, wType, dist) 
        weit_c = Kernel.GWR_W(coords, c, wType, dist)
        weit_d = Kernel.GWR_W(coords, d, wType, dist)
        
        # 3.2 decide whether local model or mixed model
        if x_glob is None: # local model
            #if  mType == 0: #mType == 0 or
                #gwrMod_a = GWR_Gaussian_Base(y, x_loc, weit_a)
                #gwrMod_b = GWR_Gaussian_Base(y, x_loc, weit_b)
                #gwrMod_c = GWR_Gaussian_Base(y, x_loc, weit_c)
                #gwrMod_d = GWR_Gaussian_Base(y, x_loc, weit_d)                
            #else:
            gwrMod_a = GWGLM_Base(y, x_loc, weit_a, mType, y_off)
            gwrMod_b = GWGLM_Base(y, x_loc, weit_b, mType, y_off)
            gwrMod_c = GWGLM_Base(y, x_loc, weit_c, mType, y_off)
            gwrMod_d = GWGLM_Base(y, x_loc, weit_d, mType, y_off)                
        else: # mixed model
            gwrMod_a = semiGWR_Base(y, x_glob, x_loc, weit_a, mType, y_off)
            gwrMod_b = semiGWR_Base(y, x_glob, x_loc, weit_b, mType, y_off)
            gwrMod_c = semiGWR_Base(y, x_glob, x_loc, weit_c, mType, y_off)
            gwrMod_d = semiGWR_Base(y, x_glob, x_loc, weit_d, mType, y_off)
            
        
        # 3.3 get diagnostic value(0: AICc, 1: AIC, 2: BIC, 3: CV)   
        if mType == 0:#or mType == 3
            f_a = getDiag_GWR[criterion](gwrMod_a)
            f_b = getDiag_GWR[criterion](gwrMod_b)
            f_c = getDiag_GWR[criterion](gwrMod_c)
            f_d = getDiag_GWR[criterion](gwrMod_d)
        else:
            f_a = getDiag_GWGLM[criterion](gwrMod_a)
            f_b = getDiag_GWGLM[criterion](gwrMod_b)
            f_c = getDiag_GWGLM[criterion](gwrMod_c)
            f_d = getDiag_GWGLM[criterion](gwrMod_d) 
        
        #print "a: %.3f, b: %.3f, c: %.3f, d: %.3f" % (a, b, c, d)             
        
        # determine next triple
        if f_b <= f_d:
            # current optimal bandwidth
            opt_weit = weit_b
            opt_band = b
            opt_cri = f_b
            c = d
            d = b
            b = a + lamda * abs(c-a)            
            if wType == 1 or wType == 3: # bandwidth is nn
                b = round(b,0)             
        else:
            # current optimal bandwidth
            opt_weit = weit_d
            opt_band = d
            opt_cri = f_d
            a = b
            b = d  
            d = c - lamda * abs(c-a)            
            if wType == 1 or wType == 3: # bandwidth is nn  
                d = round(d,0) 
            
            
        output.append((opt_band,opt_cri))
        
        # determine diff
        diff = f_b - f_d #opt_cri - pre_opt
        pre_opt = opt_cri   
        #print "diff: %.6f" % (diff)
        
    return opt_band, opt_weit, output

# to do: check whether maxVal and minVal is valid    
def f_Interval(y, x_glob, x_loc, y_off, coords, mType, wType, criterion, maxVal, minVal, interval,flag=0):
    """
    Interval search, using interval as stepsize
    
    Arguments
    ----------
        y              : array
                         n*1, dependent variable.
        x_glob         : array
                         n*k1, fixed independent variable.
        x_local        : array
                         n*k2, local independent variable, including constant.
        y_off          : array
                         n*1, offset variable for Poisson model
        coords         : dictionary
                         including (x,y) coordinates involved in the weight evaluation (including point i)  
        mType          : integer
                         GWR model type, 0: M_Gaussian, 1: M_Poisson, 2: Logistic
        wType          : integer
                         kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare 
        criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
        maxVal         : float
                         maximum value used in bandwidth searching
        minVal         : float
                         minimum value used in bandwidth searching
        interval       : float
                         interval used in interval search 
        flag           : integer
                         distance type
    Return:
           opt_band   : float
                        optimal bandwidth
           opt_weit   : kernel
                        optimal kernel
           output     : list of tuple
                        report searching process, keep bandwidth and score, [(bandwidth, score),(bandwidth, score),...]
    """
    dist = Kernel.get_pairDist(coords,flag=0) #get pairwise distance between points
    
    a = minVal
    c = maxVal
    
    # add codes to check whether a and c are valid
    #------------------------------------------------------------
    
    if wType == 1 or wType == 3: # bandwidth is nn
        a = int(a)
        c = int(c)        
    
    output = []    
   
    # 1 get initial b value
    b = a + interval #distance or nn based on wType
    if wType == 1 or wType == 3: # bandwidth is nn
        b = int(b) 
            
    # 2 create weight
    weit_a = Kernel.GWR_W(coords, a, wType, dist)  
    weit_c = Kernel.GWR_W(coords, c, wType, dist) 
    
    # 3 create model
    if x_glob is None: # local model
        #if mType == 3:
            #gwrMod_a = GWR_Gaussian(y, x_loc, weit_a)            
            #gwrMod_c = GWR_Gaussian(y, x_loc, weit_c)            
        #else:
        gwrMod_a = GWGLM_Base(y, x_loc, weit_a, mType, y_off)               
        gwrMod_c = GWGLM_Base(y, x_loc, weit_c, mType, y_off)               
    else: # mixed model
        gwrMod_a = semiGWR_Base(y, x_glob, x_loc, weit_a, mType, y_off)           
        gwrMod_c = semiGWR_Base(y, x_glob, x_loc, weit_c, mType, y_off)           
    
    # 4 get diagnostic value
    if mType == 0:#or mType == 3
        f_a = getDiag_GWR[criterion](gwrMod_a)        
        f_c = getDiag_GWR[criterion](gwrMod_c)        
    else:
        f_a = getDiag_GWGLM[criterion](gwrMod_a)        
        f_c = getDiag_GWGLM[criterion](gwrMod_c)       
    
    # 5 add to the output
    output.append((a,f_a))
    output.append((c,f_c))
    
    #print "bandwidth: %.3f, f value: %.6f" % (a, f_a)    
    #print "bandwidth: %.3f, f value: %.6f" % (c, f_c)
    
    if f_a < f_c:
        opt_weit = weit_a
        opt_band = a
        opt_val = f_a
    else:
        opt_weit = weit_c
        opt_band = c   
        opt_val = f_c
    
    while b < c:             
           
        # model using bandwidth b
        weit_b = Kernel.GWR_W(coords, b, wType, dist) # local model
        if x_glob is None: # local model                      
            #if mType == 3:
                #gwrMod_b = GWR_Gaussian(y, x_loc, weit_b)                      
            #else:
            gwrMod_b = GWGLM_Base(y, x_loc, weit_b, mType, y_off)                             
        else: # mixed model
            gwrMod_b = semiGWR_Base(y, x_glob, x_loc, weit_b, mType, y_off)           
        
        if mType == 0:#or mType == 3
            f_b = getDiag_GWR[criterion](gwrMod_b)           
        else:
            f_b = getDiag_GWGLM[criterion](gwrMod_b)        
            
        #print "bandwidth: %.3f, f value: %.6f" % (b, f_b)
        
        # add output
        output.append((b,f_b))
        
        # determine next triple
        if f_b < opt_val:
            opt_weit = weit_b
            opt_band = b
            opt_val = f_b
                    
        # update b
        b = b + interval
        
    return opt_band,opt_weit, output


def Band_Sel(y, x_glob, x_loc, coords, mType=0, y_off=None, wType=3,\
                 criterion=0, method=0, maxVal=0.0, minVal=0.0, interval=0.0, tol=1.0e-6, maxIter=200, flag=0):
    """
    Select bandwidth for kernel
    
    Methods: Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments
    ----------
        y              : array
                         n*1, dependent variable.
        x_glob         : array
                         n*k1, fixed independent variable.
        x_loc          : array
                         n*k2, local independent variable, including constant.
        coords         : dictionary
                         including (x,y) coordinates involved in the weight evaluation (including point i)  
        mType          : integer
                         GWR model type, 0: Gaussian, 1: Poisson, 2: Logistic 
        y_off          : array
                         n*1, offset variable for Poisson model
        wType          : integer
                         kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare 
        criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
        method         : integer
                         bandwidth searching method, 0: Golden, 1: Interval
        maxVal         : float
                         maximum value used in bandwidth searching
        minVal         : float
                         minimum value used in bandwidth searching
        interval       : float
                         interval used in interval search 
        tol            : float
                         tolerance used to determine convergence   
        maxIter        : integer
                         maximum number of iteration if convergence cannot arrived to the tolerance
        flag           : integer
                         distance type
    ----------
    """
    if method == 0:
        return f_Golden(y, x_glob, x_loc, y_off, coords, mType, wType, criterion, maxVal, minVal, tol, maxIter,flag)
    if method == 1:
        return f_Interval(y, x_glob, x_loc, y_off, coords, mType, wType, criterion, maxVal, minVal, interval,flag)  
    
        
    

if __name__ == '__main__': 
    
    pass

