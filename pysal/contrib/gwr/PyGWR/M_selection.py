# Author: Jing Yao
# August, 2013
# Univ. of St Andrews, Scotland, UK

# For Model estimation
import FileIO
import Kernel
import numpy as np
#from M_Gaussian import GWR_Gaussian_Base
from M_GWGLM import GWGLM_Base
from M_semiGWR import semiGWR_Base
from Diagnostics import get_AICc_GWR, get_AIC_GWR, get_BIC_GWR, get_CV_GWR, get_AICc_GWGLM, get_AIC_GWGLM, get_BIC_GWGLM
#import random

#get_kernel = {0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare} # define kernel function
getDiag_GWR = {0: get_AICc_GWR,1:get_AIC_GWR, 2:get_BIC_GWR,3: get_CV_GWR} # bandwidth selection criteria
getDiag_GWGLM = {0: get_AICc_GWGLM,1:get_AIC_GWGLM, 2:get_BIC_GWGLM}
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
    
    # Examples
    
    ##----------------------------1 bandwidth search for basic Gaussian model--------------------------------
    # read data
    flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
    flds = ['X', 'Y', 'PctBach','PctEld', 'PctFB', 'PctPov', 'PctBlack']  
    allData = FileIO.read_FILE[1](flePath)
    dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    nobs = len(dic_data.keys()) # number of observations    
    
    lst_data = []
    coords = {}
    for i in range(nobs):
        coords[i] = dic_data[i][:2] # get coordinates
        lst_data.append(dic_data[i][2:])
    arr_data = np.array(lst_data)   
               
    # create x, y    
    y = np.reshape(arr_data[:,0], (-1,1))
    x = arr_data[:,1:]
    x = np.hstack((np.ones(y.shape),x)) 
   
    #----1. bandwidth selection
    # bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
    cri = ["AICc", "AIC", "BIC", "CV"]
    # kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare
    rs = Band_Sel(y, None, x, coords, mType=0, y_off=None, wType=3,\
                 criterion=0, method=0)
    #rs = Band_Sel(y, None, x, coords, mType=0, wType=3,\
                 #criterion=0, method=1, maxVal=130.0, minVal=50.0, interval=4.0, tol=1.0e-6)
    
    print "ok!"
    print "optimal bandwidth: %.3f" % (rs[0])
    for elem in rs[2]:
        print "bandwidth: %.3f, %5s: %.6f" % (elem[0], cri[0], elem[1])
        
##-------------------------------------------Golden search------------------------------------------------------------    
        
    ##*****************************1 output for bandwidth choice: basic Gaussian, AICc, adaptive bisquare*************************
    ##**********************************************************************************************************
    
    #ok!
    #optimal bandwidth: 129.000
    #bandwidth: 127.000, f value: 937.688083
    #bandwidth: 127.000, f value: 937.688083
    #bandwidth: 127.000, f value: 937.688083
    #bandwidth: 127.000, f value: 937.688083
    #bandwidth: 127.000, f value: 937.688083
    #bandwidth: 129.000, f value: 937.620032
    #bandwidth: 129.000, f value: 937.620032
    #bandwidth: 129.000, f value: 937.620032 
    
    # results from GWR4, using AICc, adaptive bisquare
    
    #Bandwidth search <golden section search>
    #Limits: 50,  174
    #Golden section search begins...
    #Initial values
    #pL            Bandwidth:    50.000 Criterion:    953.379
    #p1            Bandwidth:    97.364 Criterion:    940.459
    #p2            Bandwidth:   126.636 Criterion:    937.852
    #pU            Bandwidth:   174.000 Criterion:    947.329
    #iter    1 (p2) Bandwidth:   126.636 Criterion:    937.852 Diff:     29.272
    #iter    2 (p1) Bandwidth:   126.636 Criterion:    937.852 Diff:     18.091
    #iter    3 (p2) Bandwidth:   126.636 Criterion:    937.852 Diff:     11.181
    #iter    4 (p1) Bandwidth:   126.636 Criterion:    937.852 Diff:      6.910
    #iter    5 (p2) Bandwidth:   126.636 Criterion:    937.852 Diff:      4.271
    #iter    6 (p2) Bandwidth:   129.276 Criterion:    937.620 Diff:      2.639
    #iter    7 (p1) Bandwidth:   129.276 Criterion:    937.620 Diff:      1.631
    #Best bandwidth size 129.000
    #Minimum AICc      937.620    
    
    ##*****************************2 output for bandwidth choice: basic Gaussian, AICc, fixed Gaussian  *************************
    ##**********************************************************************************************************
    #optimal bandwidth: bandwidth: 93162.482, f value: 935.063044 
    # results from GWR4: 
    # Best bandwidth size 93841.272
    # Minimum AICc      935.065
   
    
    ##*****************************3 output for bandwidth choice: basic Gaussian, cv, adaptive bisquare (cv only for Gaussian model)*************************
    ##**********************************************************************************************************
    
    #ptimal bandwidth: 147.000
    #bandwidth: 127.000, f value: 14.495299
    #bandwidth: 145.000, f value: 14.335115
    #bandwidth: 145.000, f value: 14.335115
    #bandwidth: 145.000, f value: 14.335115
    #bandwidth: 145.000, f value: 14.335115
    #bandwidth: 145.000, f value: 14.335115
    #bandwidth: 146.000, f value: 14.333277
    #bandwidth: 147.000, f value: 14.328231
    #bandwidth: 147.000, f value: 14.328231
    #bandwidth: 147.000, f value: 14.328231

    
    # results from GWR4
    # 144, cv: 14.332
    
##-------------------------------------------Interval search------------------------------------------------------------ 
    
    ##*****************************1 output for bandwidth choice: AICc, adaptive bisquare*************************
    ##**********************************************************************************************************
    #optimal bandwidth: 130
    #bandwidth d: 50.000, f value: 953.379396
    #bandwidth d: 130.000, f value: 937.773613
    #bandwidth b: 54.000, f value: 951.974380
    #bandwidth b: 58.000, f value: 949.594479
    #bandwidth b: 62.000, f value: 947.621671
    #bandwidth b: 66.000, f value: 946.142938
    #bandwidth b: 70.000, f value: 946.213416
    #bandwidth b: 74.000, f value: 944.929880
    #bandwidth b: 78.000, f value: 943.988151
    #bandwidth b: 82.000, f value: 942.879626
    #bandwidth b: 86.000, f value: 941.981403
    #bandwidth b: 90.000, f value: 941.831904
    #bandwidth b: 94.000, f value: 940.777698
    #bandwidth b: 98.000, f value: 939.972137
    #bandwidth b: 102.000, f value: 939.842204
    #bandwidth b: 106.000, f value: 939.059813
    #bandwidth b: 110.000, f value: 938.039129
    #bandwidth b: 114.000, f value: 938.605162
    #bandwidth b: 118.000, f value: 938.190623
    #bandwidth b: 122.000, f value: 938.106641
    #bandwidth b: 126.000, f value: 937.851743
    
    #results from GWR4
    #Bandwidth search <interval search> min, max, step
    #50,  130,  4
    #Bandwdith:   130.000  Dev:    904.362  trace(Hat):   14.155  Criterion:    937.774 Valid_fit
    #Bandwdith:   126.000  Dev:    903.279  trace(Hat):   14.634  Criterion:    937.852 Valid_fit
    #Bandwdith:   122.000  Dev:    902.313  trace(Hat):   15.134  Criterion:    938.107 Valid_fit
    #Bandwdith:   118.000  Dev:    900.961  trace(Hat):   15.719  Criterion:    938.191 Valid_fit
    #Bandwdith:   114.000  Dev:    899.951  trace(Hat):   16.295  Criterion:    938.605 Valid_fit
    #Bandwdith:   110.000  Dev:    897.860  trace(Hat):   16.907  Criterion:    938.039 Valid_fit
    #Bandwdith:   106.000  Dev:    897.124  trace(Hat):   17.605  Criterion:    939.060 Valid_fit
    #Bandwdith:   102.000  Dev:    895.971  trace(Hat):   18.368  Criterion:    939.842 Valid_fit
    #Bandwdith:    98.000  Dev:    894.001  trace(Hat):   19.187  Criterion:    939.972 Valid_fit
    #Bandwdith:    94.000  Dev:    892.483  trace(Hat):   20.083  Criterion:    940.778 Valid_fit
    #Bandwdith:    90.000  Dev:    891.041  trace(Hat):   21.034  Criterion:    941.832 Valid_fit
    #Bandwdith:    86.000  Dev:    888.308  trace(Hat):   22.117  Criterion:    941.981 Valid_fit
    #Bandwdith:    82.000  Dev:    886.365  trace(Hat):   23.170  Criterion:    942.880 Valid_fit
    #Bandwdith:    78.000  Dev:    884.328  trace(Hat):   24.318  Criterion:    943.988 Valid_fit
    #Bandwdith:    74.000  Dev:    881.467  trace(Hat):   25.683  Criterion:    944.930 Valid_fit
    #Bandwdith:    70.000  Dev:    878.371  trace(Hat):   27.224  Criterion:    946.213 Valid_fit
    #Bandwdith:    66.000  Dev:    873.531  trace(Hat):   28.866  Criterion:    946.143 Valid_fit
    #Bandwdith:    62.000  Dev:    869.916  trace(Hat):   30.578  Criterion:    947.622 Valid_fit
    #Bandwdith:    58.000  Dev:    865.014  trace(Hat):   32.826  Criterion:    949.594 Valid_fit
    #Bandwdith:    54.000  Dev:    860.258  trace(Hat):   35.084  Criterion:    951.974 Valid_fit
    #Bandwdith:    50.000  Dev:    852.765  trace(Hat):   37.800  Criterion:    953.379 Valid_fit
    #Best bandwidth size 130.000
    #Minimum AICc      937.774
    
    ##-------------------------------------------2 bandwidth search for Poisson model-------------------------
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt" #"E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
    #flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OCC_TEC', 'OWNH', 'POP65', 'UNEMP']  
    #allData = FileIO.read_FILE[3](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    ## reformat data to float
    #for key, val in dic_data.items():
        #dic_data[key] = tuple(float(elem) for elem in val)
    
    #lst_data = []
    #coords = {}
    #for i in range(nobs):
        #coords[i] = dic_data[i][:2] # get coordinates
        #lst_data.append(dic_data[i][2:])
    #arr_data = np.array(lst_data)   
               
    ## create x, y    
    #y = np.reshape(arr_data[:,0], (-1,1))
    #y_off = np.reshape(arr_data[:,1], (-1,1))
    #x = arr_data[:,2:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    #band = 84
    #weit = Kernel.GWR_W(coords, band, 3)   
    #myMod = GWGLM_Base(y, x, weit, 1, y_off)
    #print getDiag_GWGLM[0](myMod)
    
    ## bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
    #cri = ["AICc", "AIC", "BIC", "CV"]
    ## kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare
    #rs = Band_Sel(y, None, x, coords, 1, y_off, wType=0, criterion=0, method=0)
    ##rs = Band_Sel(y, None, x, coords, mType=3, wType=3,\
                 ##criterion=0, method=1, maxVal=130.0, minVal=50.0, interval=4.0, tol=1.0e-6)
    
    #print "ok!"
    #print "optimal bandwidth: %.3f" % (rs[0])
    #for elem in rs[2]:
        #print "bandwidth: %.3f, %5s: %.6f" % (elem[0], cri[0], elem[1])
        
##---------------------golden search: Poisson, -------------------------------------------------------------
    #1----------------------------------- AICc, adaptive bisquare--------------------------------------
    #optimal bandwidth: 95.000
    #bandwidth: 131.000,  AICc: 373.550091
    #bandwidth: 100.000,  AICc: 367.110279
    #bandwidth: 81.000,  AICc: 366.207635
    #bandwidth: 81.000,  AICc: 366.207635
    #bandwidth: 88.000,  AICc: 366.193456
    #bandwidth: 93.000,  AICc: 365.878448
    #bandwidth: 95.000,  AICc: 365.472758
    #bandwidth: 95.000,  AICc: 365.472758
    #bandwidth: 95.000,  AICc: 365.472758
    
    # results from GWR4
    # Best bandwidth size 84.000
    # Minimum AICc      365.593
    
    #2----------------------------------- AICc, fixed Gaussian-------------------------------------------
    #optimal bandwidth: 16527.163
    #bandwidth: 31134.909,  AICc: 386.239951
    #bandwidth: 22590.074,  AICc: 375.208850
    #bandwidth: 17309.309,  AICc: 367.861935
    #bandwidth: 17309.309,  AICc: 367.861935
    #bandwidth: 17309.309,  AICc: 367.861935
    #bandwidth: 16062.562,  AICc: 367.732306
    #bandwidth: 16062.562,  AICc: 367.732306
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16538.811,  AICc: 367.647408
    #bandwidth: 16528.639,  AICc: 367.647355
    #bandwidth: 16528.639,  AICc: 367.647355
    #bandwidth: 16528.639,  AICc: 367.647355
    #bandwidth: 16526.265,  AICc: 367.647354
    #bandwidth: 16526.265,  AICc: 367.647354
    #bandwidth: 16527.163,  AICc: 367.647354

    # results from GWR4
    # Best bandwidth size 16538.660
    # Minimum AICc      367.647
    
    ##-------------------------------------------3 bandwidth search for mixed model-------------------------
    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt" #"E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
    #flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OCC_TEC', 'OWNH', 'POP65', 'UNEMP']  
    #allData = FileIO.read_FILE[3](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    ## reformat data to float
    #for key, val in dic_data.items():
        #dic_data[key] = tuple(float(elem) for elem in val)
    
    #lst_data = []
    #coords = {}
    #for i in range(nobs):
        #coords[i] = dic_data[i][:2] # get coordinates
        #lst_data.append(dic_data[i][2:])
    #arr_data = np.array(lst_data)   
               
    ## create x, y    
    #y = np.reshape(arr_data[:,0], (-1,1))
    #y_off = np.reshape(arr_data[:,1], (-1,1))
    #x_local = arr_data[:,2:4]
    #x_local = np.hstack((np.ones(y.shape),x_local))    
    #x_global = arr_data[:,4:]
    
    ## bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
    #cri = ["AICc", "AIC", "BIC", "CV"]
    ## kernel type, 0: fix_Gaussian, 1: adap_Gaussian, 2: fix_Bisquare, 3: adap_Bisquare
    #rs = Band_Sel(y, x_global, x_local, coords, 1, y_off, wType=0, criterion=0, method=0)
    
    #print "ok!"
    #print "optimal bandwidth: %.3f" % (rs[0])
    #for elem in rs[2]:
        #print "bandwidth: %.3f, %5s: %.6f" % (elem[0], cri[0], elem[1])
    
    ##---------------------golden search: semiPoisson---------------------------------------------------------------
    #1----------------------------------- AICc, adaptive bisquare--------------------------------------
    #optimal bandwidth: 181.000
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    #bandwidth: 181.000,  AICc: 391.995013
    
    # results from GWR4
    # Best bandwidth size 178.000
    # Minimum AICc      391.963
    
    #2----------------------------------- AICc, fixed Gaussian-------------------------------------------
    #optimal bandwidth: 20432.618
    #bandwidth: 31134.909,  AICc: 391.294221
    #bandwidth: 22590.074,  AICc: 382.104802
    #bandwidth: 22590.074,  AICc: 382.104802
    #bandwidth: 22590.074,  AICc: 382.104802
    #bandwidth: 20573.104,  AICc: 380.572522
    #bandwidth: 20573.104,  AICc: 380.572522
    #bandwidth: 20573.104,  AICc: 380.572522
    #bandwidth: 20573.104,  AICc: 380.572522
    #bandwidth: 20573.104,  AICc: 380.572522
    #bandwidth: 20391.137,  AICc: 380.563475
    #bandwidth: 20391.137,  AICc: 380.563475
    #bandwidth: 20460.676,  AICc: 380.563194
    #bandwidth: 20460.676,  AICc: 380.563194
    #bandwidth: 20434.094,  AICc: 380.562734
    #bandwidth: 20434.094,  AICc: 380.562734
    #bandwidth: 20434.094,  AICc: 380.562734
    #bandwidth: 20434.094,  AICc: 380.562734
    #bandwidth: 20434.094,  AICc: 380.562734
    #bandwidth: 20431.720,  AICc: 380.562732
    #bandwidth: 20431.720,  AICc: 380.562732
    #bandwidth: 20432.618,  AICc: 380.562732


    # results from GWR4
    # Best bandwidth size 20391.016
    # Minimum AICc      380.562
    
    ##---------------------golden search: semiGaussian---------------------------------------------------------------
    #flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
    #flds = ['X', 'Y', 'PctBach','PctEld','PctPov', 'PctFB', 'PctBlack']  
    #allData = FileIO.read_FILE[1](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    #lst_data = []
    #coords = {}
    #for i in range(nobs):
        #coords[i] = dic_data[i][:2] # get coordinates
        #lst_data.append(dic_data[i][2:])
    #arr_data = np.array(lst_data)   
               
    ## create x, y    
    #y = np.reshape(arr_data[:,0], (-1,1))
    #x_local = arr_data[:,3:]
    #x_global = arr_data[:,1:3] 
    #x_global = np.hstack((np.ones(y.shape),x_global)) 
     
    #rs = Band_Sel(y, x_global, x_local, coords, 0)
    #cri = ["AICc", "AIC", "BIC", "CV"]
    #print "ok!"
    #print "optimal bandwidth: %.3f" % (rs[0])
    #print rs[1].band
    #for elem in rs[2]:
        #print "bandwidth: %.3f, %5s: %.6f" % (elem[0], cri[0], elem[1])
        
    ##----------output----------------
    #optimal bandwidth: 129.000
    #bandwidth: 127.000,  AICc: 941.735023
    #bandwidth: 127.000,  AICc: 941.735023
    #bandwidth: 127.000,  AICc: 941.735023
    #bandwidth: 127.000,  AICc: 941.735023
    #bandwidth: 127.000,  AICc: 941.735023
    #bandwidth: 129.000,  AICc: 941.627955
    #bandwidth: 129.000,  AICc: 941.627955
    #bandwidth: 129.000,  AICc: 941.627955

    
    