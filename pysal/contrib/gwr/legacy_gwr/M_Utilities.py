# Author: Jing Yao
# December, 2013
# Univ. of St Andrews, Scotland, UK

# For some optional functions:
# 1 prediction at non-sample points
# 2 variable standardization
# 3 geographical variability test
# 4 L->G variable selection
# 5 G->L variable selection

#----for references--------------------------------------------------------------------------------------------
# get_criteria = {0: get_AICc_GWR, 1:get_AIC_GWR, 2:get_BIC_GWR, 3: get_CV_GWR} # bandwidth selection criteria
# get_GWRModel = {0: GWR_Gaussian_Base, 1: GWR_Poisson_Base, 2: GWR_Logistic_Base} # GWR model type

import numpy as np
import Kernel
from M_GWGLM import GWGLM_Base
#from M_Gaussian import GWR_Gaussian_Base
from M_semiGWR import semiGWR_Base
#from M_OLS import OLS_Base
from M_GLM import GLM_Base
import FileIO
import M_selection
import Diagnostics
from datetime import datetime

getDiag_GWR = {0: Diagnostics.get_AICc_GWR,1:Diagnostics.get_AIC_GWR, 2:Diagnostics.get_BIC_GWR,3: Diagnostics.get_CV_GWR}
getDiag_GWGLM = {0: Diagnostics.get_AICc_GWGLM,1:Diagnostics.get_AIC_GWGLM, 2:Diagnostics.get_BIC_GWGLM}
#getDiag_OLS = {0: Diagnostics.get_AICc_OLS,1:Diagnostics.get_AIC_OLS, 2:Diagnostics.get_BIC_OLS,3: Diagnostics.get_CV_OLS}
getDiag_GLM = {0: Diagnostics.get_AICc_GLM,1:Diagnostics.get_AIC_GLM, 2:Diagnostics.get_BIC_GLM,3: Diagnostics.get_CV_OLS}

def StandardVars(xVars):
    """
    Standardize independent variables: 
    z-transformation so that each variable has zero mean and one standard deviation
    z = (x - mean)/sigma
    
    Arguments:
        xVars         : array, 
                        n*k, including independent variables
            
    Return:
        s_xVars       : array,
                        n*k, including stadardized variables    
    """
    nVars  =  len(xVars[0])
    s_xVars = np.ones(xVars.shape)
    for i in range(nVars):
        xMean = np.mean(xVars[:,i])
        xSigma = np.std(xVars[:,i])
        s_xVars[:,i] = (xVars[:,i] - xMean) * 1.0 / xSigma

    return s_xVars

def varyTest(y, x_glob, x_loc, kernel, mType=0, y_off=None, criterion=0, orig_mod=None):
    """
    Geographical variability test
    All the models use the same bandwidth
    
    Arguments
    ----------
        y              : array
                         n*1, dependent variable.
        x_glob         : array
                         n*k1, fixed independent variable.
        x_local        : array
                         n*k2, local independent variable, including constant.
        kernel         : dictionary
                         including (x,y) coordinates involved in the weight evaluation (including point i)  
        mType          : integer
                         GWR model type, 0: Gaussian, 1: Poisson, 2: Logistic
        y_off          : array
                         n*1, offset variable for Poisson model
	criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
	orig_mod       : object of GWR model
	                 original model
    Return:
        geoVary        : list of tuple
			 including each X variable and associated F statistics: [(x1,f_stat),(x2,f_stat),...()]
    
    """
    nObs = len(y)    
        
    # 1 original model
    if orig_mod is None:
	if x_glob is None:
	    gwrMod_old = GWGLM_Base(y, x_loc, kernel, mType, y_off)
	else:
	    gwrMod_old = semiGWR_Base(y,x_glob,x_loc,kernel,mType,y_off)
    else:
	gwrMod_old = orig_mod
    
    # 2 original statistics
    p_old = gwrMod_old.tr_S      
    if mType == 0 : 
	cri_old = getDiag_GWR[criterion](gwrMod_old)
	dev_old = gwrMod_old.res2
    else:
	cri_old = getDiag_GWGLM[criterion](gwrMod_old)	
	dev_old = gwrMod_old.dev_res
    
    # 3 loop
    geoVary = []
    nVar_loc = len(x_loc[0])
    if nVar_loc > 0: # intercept is local variable by default
	for i in range(nVar_loc):	    
	    tmpX = x_loc[:,i] 
	    # new x_loc
	    xg = np.delete(x_loc, i, 1)
	    # new x_glob
	    if x_glob is None:
		xf = tmpX
		xf = np.reshape(xf,(-1,1))
	    else:
		nVar_glob = len(x_glob[0])
		xf = np.zeros(shape=(nObs,nVar_glob))
		xf = x_glob
		tmpX = np.reshape(tmpX,(-1,1))
		xf = np.hstack((xf,tmpX))
	    # new model
	    gwrMod_new = semiGWR_Base(y,xf,xg,kernel,mType,y_off)
	    # new statistics
	    p_new = gwrMod_new.tr_S 	    
	    if mType == 0 : 
		cri_new = getDiag_GWR[criterion](gwrMod_new)
		dev_new = gwrMod_new.res2
	    else:
		cri_new = getDiag_GWGLM[criterion](gwrMod_new)		
		dev_new = gwrMod_new.dev_res
	    # differentce
	    diffp = p_old - p_new
	    diffcri = cri_old - cri_new
	    if mType == 0: 		
		df = nObs - p_old
		f = ( (dev_new - dev_old) / diffp ) / ( dev_old/ df )		
		geoVary.append((f,diffp, df,diffcri))
	    else:    
		diffdev = dev_old - dev_new		
		geoVary.append((diffdev,diffp,diffcri))
	    
    return geoVary

def L2G(y, x_glob, x_loc, coords, mType=0, wType=3, y_off=None, orig_mod=None, criterion=0, bdinfo=0, band=0, maxVal=0.0, minVal=0.0, interval=0.0, tol=1.0e-2, maxIter=50):
    """
    Variable selection: local to global
    
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
        wType          : integer
                         weight type  
        mType          : integer
                         GWR model type, 0: Gaussian, 1: Poisson, 2: Logistic
        y_off          : array
                         n*1, offset variable for Poisson model
	orig_mod       : object of GWR model
	                 original model
	criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
	bdinfo         : integer
	                 bandwidth searching method: 0: golden search 1: interval 2: fixed single bandwidth
	band           : float
	                 given bandwidth if bdinfo=2 
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
               
            
    Return:
        varsL          : list,
                         ids of local Xs 
        varsG          : list,
                         ids of global Xs 
	optband        : list
	                 info of optimal bandwidth searching results
	optWeit        : GWR kernel
	                 kernel of the best model
    """
    nObs = len(y)
    nVars_loc = len(x_loc[0])  
    if x_glob is None:
	nVars_glob = 0
	tmp_glob = np.zeros(shape=(nObs,0))
    else:
	nVars_glob = len(x_glob[0])
	tmp_glob = np.zeros(shape=(nObs,nVars_glob))
    nVars = nVars_loc + nVars_glob
    optband = []
    
    # 1 original model
    # check original bandwidth
    if orig_mod is None:
	if bdinfo == 0 or bdinfo == 1: # golden or interval search
	    rs = M_selection.Band_Sel(y, x_glob, x_loc, coords, mType, y_off, wType, criterion, bdinfo, maxVal, minVal, interval, tol, maxIter)
	    band = rs[0]
	    weit = rs[1]
	    #optband.append("Golden section starts:")
	    optband.append(rs)
	else:	    
	    # set original kernel
	    weit = Kernel.GWR_W(coords, band, wType)   
	
	# set original model
	if x_glob is None:
	    gwrMod_old = GWGLM_Base(y, x_loc, weit, mType, y_off)
	else:
	    gwrMod_old = semiGWR_Base(y,x_glob,x_loc,weit,mType,y_off)
    else:
	weit = orig_mod.kernel
	gwrMod_old = orig_mod
	
    optWeit = weit   
	
    # get original diagnostics
    if mType == 0:
	cri_old = getDiag_GWR[criterion](gwrMod_old)
    else:
	cri_old = getDiag_GWGLM[criterion](gwrMod_old)
    
    #print "original cri:"
    #print cri_old
    
    # 2 loop    
    flag = True # check whether is x moved to global
    if nVars_loc > 0:
	orilist = range(nVars_loc) # ids of original local Xs			
	while flag: #  until no improvement in one loop in orilist
	    flag = False
	    #print "original list:"
	    #print orilist
	    outlist = [] # ids of Xs from local to global
	    n_currXs = len(orilist) # every time loop through orilist
	    # set local x
	    tmp_loc = np.zeros(shape=(nObs,0))
	    for i in orilist:
		tmp_loc = np.hstack((tmp_loc,np.reshape(x_loc[:,i],(-1,1))))	
	    for i in range(n_currXs):
		idx = orilist[i]
		#print i
		#print idx
		# try to remove ith x
		x_out = np.reshape(x_loc[:,idx],(-1,1))
		tmp_loc = np.delete(tmp_loc, i-len(outlist), 1)
		# get new x_glob
		tmp_glob = np.hstack((tmp_glob,x_out))
		# decide whether is a global model
		if len(tmp_glob[0]) == nVars:   # global model
		    gwrMod_new = GLM_Base(y,tmp_glob,mType,y_off) # use GLM		    
		    cri_new = getDiag_GLM[criterion](gwrMod_new)
		else:            # should be mixed model
		    # new bandwidth
		    if bdinfo == 0 or bdinfo == 1: # golden or interval search
			rs = M_selection.Band_Sel(y, tmp_glob, tmp_loc, coords, mType, y_off, wType, criterion, bdinfo, maxVal, minVal, interval, tol, maxIter)
			band = rs[0]
			weit = rs[1]			
			optband.append(rs)
		    else:
			# new kernel
			weit = Kernel.GWR_W(coords, band, wType) 
		    
		    # new model
		    gwrMod_new = semiGWR_Base(y,tmp_glob,tmp_loc,weit,mType,y_off)
		    # get diagnostics
		    if mType == 0:
			cri_new = getDiag_GWR[criterion](gwrMod_new)
		    else:
			cri_new = getDiag_GWGLM[criterion](gwrMod_new)
		#print cri_new
		# check improvements
		if cri_new < cri_old: # move x from local to global
		    outlist.append(idx)
		    cri_old = cri_new # update criteria
		    optWeit = weit
		    flag = True
		else:
		    tmp_glob = np.delete(tmp_glob, -1, 1) # move x back to local
		    tmp_loc = np.hstack((x_out,tmp_loc))
	    orilist = list(set(orilist) - set(outlist))
	    #print "outlist:"
	    #print outlist
	#print "old cri:"
	#print cri_old
	
    varsL = orilist
    varsG = list(set(range(nVars_loc)) - set(orilist))
    
    
    return varsL, varsG, optband, optWeit, cri_old

def G2L(y, x_glob, x_loc, coords, mType=0, wType=3, y_off=None, orig_mod=None, criterion=0, bdinfo=0, band=0, maxVal=0.0, minVal=0.0, interval=0.0, tol=1.0e-2, maxIter=50):
    """
    Variable selection: global to local
    
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
        wType          : integer
                         weight type  
        mType          : integer
                         GWR model type, 0: Gaussian, 1: Poisson, 2: Logistic
        y_off          : array
                         n*1, offset variable for Poisson model
	orig_mod       : object of GWR model
	                 original model
	criterion      : integer
                         bandwidth selection criterion, 0: AICc, 1: AIC, 2: BIC, 3: CV
	bdinfo         : integer
	                 bandwidth searching method: 0: golden search 1: interval 2: fixed single bandwidth
	band           : float
	                 given bandwidth if bdinfo=2 
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
               
            
    Return:
        varsL          : list,
                         ids of local Xs 
        varsG          : list,
                         ids of global Xs 
	optband        : list
	                 info of optimal bandwidth searching results
	optWeit        : kernel
	                 kernel of best model
	optcri         : float
		         criterion value for optimal model
    """
    nObs = len(y)
    nVars_glob = len(x_glob[0])  
    if x_loc is None:
	nVars_loc = 0
	tmp_loc = np.zeros(shape=(nObs,0))
    else:
	nVars_loc = len(x_loc[0])
	tmp_loc = np.zeros(shape=(nObs,nVars_loc))
	tmp_loc = x_loc
    nVars = nVars_loc + nVars_glob
    optband = []
    
    # loop    
    flag = True # check whether is x moved to global
    if nVars_glob > 0:    
	if orig_mod is None:
	    # 1 set original model
	    if x_loc is None:  # global model
		gwrMod_old = GLM_Base(y,x_glob,mType,y_off)
		cri_old = getDiag_GLM[criterion](gwrMod_old)
	    else:   # should be mixed model# check original bandwidth		
		if bdinfo == 0 or bdinfo == 1: # golden or interval search
		    rs = M_selection.Band_Sel(y, x_glob, x_loc, coords, mType, y_off, wType, criterion, bdinfo, maxVal, minVal, interval, tol, maxIter)
		    band = rs[0]
		    weit = rs[1]
		    optband.append(rs)
		else: 
		    # set original kernel
		    weit = Kernel.GWR_W(coords, band, wType)
		optWeit = weit
		gwrMod_old = semiGWR_Base(y,x_glob,x_loc,weit,mType,y_off)
		# get original diagnostics
		if mType == 0:
		    cri_old = getDiag_GWR[criterion](gwrMod_old)
		else:
		    cri_old = getDiag_GWGLM[criterion](gwrMod_old)
	else:
	    gwrMod_old = orig_mod
	    weit = orig_mod.kernel	    
	    optWeit = weit
	
	#print "original cri:"
	#print cri_old
    
	# 2 loop
	orilist = range(nVars_glob) # ids of original global Xs			
	while flag: #  until no improvement in one loop in orilist
	    flag = False
	    #print "original list:"
	    #print orilist
	    outlist = [] # ids of Xs from global to local
	    n_currXs = len(orilist) # every time loop through orilist
	    # set global x
	    tmp_glob = np.zeros(shape=(nObs,0))
	    for i in orilist:
		tmp_glob = np.hstack((tmp_glob,np.reshape(x_glob[:,i],(-1,1))))	
	    for i in range(n_currXs):
		idx = orilist[i]
		#print i
		#print idx
		# try to remove ith x
		x_out = np.reshape(x_glob[:,idx],(-1,1))
		tmp_glob = np.delete(tmp_glob, i-len(outlist), 1)
		# get new x_loc
		tmp_loc = np.hstack((tmp_loc,x_out))
		# new bandwidth
		if bdinfo == 0 or bdinfo == 1: # golden or interval search
		    rs = M_selection.Band_Sel(y, tmp_glob, tmp_loc, coords, mType, y_off, wType, criterion, bdinfo, maxVal, minVal, interval, tol, maxIter)
		    band = rs[0]
		    weit = rs[1]
		    optband.append(rs)
		else:
		    # new kernel
		    weit = Kernel.GWR_W(coords, band, wType) 
		
		# decide whether is a local model
		if len(tmp_loc[0]) == nVars:   # local model		   
		    gwrMod_new = GWGLM_Base(y,tmp_loc,weit,mType,y_off) 
		    cri_new = getDiag_GWGLM[criterion](gwrMod_new)
		else:            # should be mixed model
		    gwrMod_new = semiGWR_Base(y,tmp_glob,tmp_loc,weit,mType,y_off)		    
		    if mType == 0:# get diagnostics
			cri_new = getDiag_GWR[criterion](gwrMod_new)
		    else:
			cri_new = getDiag_GWGLM[criterion](gwrMod_new)
		#print cri_new
		# check improvements
		if cri_new < cri_old: # move x from local to global
		    outlist.append(idx)
		    cri_old = cri_new # update criteria
		    flag = True
		    optWeit = weit
		else:
		    tmp_loc = np.delete(tmp_loc, -1, 1) # move x back to local
		    tmp_glob = np.hstack((x_out,tmp_glob))
	    orilist = list(set(orilist) - set(outlist))
	    #print "outlist:"
	    #print outlist
	#print "old cri:"
	#print cri_old
	
    varsG = orilist
    varsL = list(set(range(nVars_glob)) - set(orilist))    
    
    return varsL, varsG, optband, optWeit, cri_old


def pred(data, refData, band, y, x_local, y_hat=None, wType=0, mType=0, flag=0, y_offset=None, sigma2=1, y_fix=None, fMatrix=None):
    """
    predict values at unsampled locations
    
    Arguments:
        data           : dictionary, 
                         (x,y) of unsampled locations
        refData        : dictionary,
                         (x,y) of sampled locations  
        band           : float
                         bandwidth
        y              : array
                         n*1, dependent variable
        y_hat          : array
                         n*1, predicted y from original model, to calculate local statistics
        x_local        : array
                         n*k1, local independent variable
        y_offset       : array
                         n*1, offset variable for Poisson model
	sigma2         : float
	                 used to calculate std. error of betas for Gaussian model
        y_fix          : array
                         n*1, fixed part of y from global Xs, used in mixed model
        fMatrix        : array
                         n*n, hat matrix for global model, used in mixed model
        wType          : integer
                         define which kernel function to use  
        mType          : integer
                         model type, model type, 0: Gaussian, 1: Poisson, 2: Logistic
        flag           : dummy,
                         0 or 1, 0: Euclidean distance; 1: spherical distance
               
            
    Return:
        Betas          : array
                         n*k, Beta estimation
        std_err        : array
                         n*k, standard errors of Beta
        t_stat         : array
                         n*k, local t-statistics
        localR2        : array
                         n*1, local R square or local p-dev   
    """
    # 1 get W matrix
    dicDist = {}
    n_pred = len(data.keys())
    for i in range(n_pred):# calculate distance between unsampled obs and sampled obs
        dicDist[i] = Kernel.get_focusDist(data[i], refData, flag)
    
    weit = Kernel.GWR_W(data, band, wType, dicDist)  
    #print len(dicDist[0].keys())
    #print len(weit.w.keys())
    #print len(weit.w[0])
    
    # 2 get predicted local Beta estimation        
    #if mType == 0:# 2.1 basic Gaussian
        #mod_loc = GWR_Gaussian_Base(y, x_local, weit)        
    #else:# 2.2 GWGLM models including mixed models
    mod_loc = GWGLM_Base(y, x_local, weit, mType, y_offset, y_fix, fMatrix)
    
    pred_betas = mod_loc.Betas[:n_pred]
        
    # 3 get std errors of Betas
    #if mType == 1 or mType == 2:
        #sigma2 = 1.0
    pred_stdErr = np.sqrt(mod_loc.CCT * sigma2)
    
    # 4 get t statistics
    pred_tstat = pred_betas/pred_stdErr
    
    # 5 get local R2 or local p-dev
    localR2 = np.zeros(shape=(n_pred,1)) 
    n_reg = len(y)
    if mType == 0 : # Gaussian model  or mType == 3
	for i in range(n_pred):
	    w_i= np.reshape(np.array(weit.w[i]), (-1, 1))
	    sum_yw = np.sum(y * w_i)
	    ybar = 1.0 * sum_yw / np.sum(w_i)
	    rss = np.sum(w_i * (y - y_hat)**2) 
	    tss = np.sum(w_i * (y - ybar)**2) 
	    localR2[i] = (tss - rss)/tss
    if mType == 1: # Poisson model
	for i in range(n_pred):
	    w_i= np.reshape(np.array(weit.w[i]), (-1, 1))
	    sum_yw = np.sum(y * w_i)
	    ybar = 1.0 * sum_yw / np.sum(w_i * y_offset)
	    dev = 0.0
	    dev0 = 0.0
	    for j in range(n_reg):
		if y[j] <>0: 
		    dev += 2 * y[j] * (np.log(y[j]) - np.log(y_hat[j])) * w_i[j]
		    dev0 += 2 * y[j] * (np.log(y[j]) - np.log(ybar * y_offset[j])) * w_i[j]
		dev -= 2 * (y[j] - y_hat[j]) * w_i[j]
		dev0 -= 2 * (y[j] - ybar * y_offset[j]) * w_i[j]		
	    localR2[i] = 1.0 - dev / dev0
    if mType == 2: # Logistic model
	for i in range(n_pred):
	    w_i= np.reshape(np.array(weit.w[i]), (-1, 1))
	    sum_yw = np.sum(y * w_i)
	    ybar = 1.0 * sum_yw / np.sum(w_i)
	    dev = 0.0
	    dev0 = 0.0
	    for j in range(n_reg):
		if (1.0 - y_hat[j] < 1e-10):
		    nu = np.log(y_hat[j]/1e-10)
		    dev += -2* (y[j] * nu + np.log(1e-10) ) * w_i[j]
		else:
		    nu = np.log(y_hat[j]/(1.0 - y_hat[j]))
		    dev += -2* (y[j] * nu + np.log(1.0 - y_hat[j])) * w_i[j]
		nu0 = np.log(ybar/(1-ybar))
		dev0 += -2* (y[j] * nu0 + np.log(1.0 - ybar) ) * w_i[j]
			
	    localR2[i] = 1.0 - dev / dev0
			
    
    return pred_betas, pred_stdErr, pred_tstat, localR2


    
if __name__ == '__main__': 
    
    ##----test standardisation variables--------------------------
    #a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #print StandardVars(a)
    ##--------------------end-------------------------------------
    
    ##----test prediction on non-regression points: GW Gaussian model----------------
    ## Examples
    
    ## 1 read regression data
    #flePath = "E:/UK work/GWR/Code/Sample data/Georgia/Prediction/Georgia_reg.txt"
    #flds = ['X', 'Y', 'PctBach','PctEld', 'PctFB', 'PctPov', 'PctBlack']  
    #allData = FileIO.read_FILE[3](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    ## change to numerical values
    #for key,val in dic_data.items():
        #dic_data[key] = [float(elem) for elem in val]
    
    #lst_data = []
    #coords = {}
    #for i in range(nobs):
        #coords[i] = dic_data[i][:2] # get coordinates
        #lst_data.append(dic_data[i][2:])
    #arr_data = np.array(lst_data)   
               
    ## create x, y    
    #y = np.reshape(arr_data[:,0], (-1,1))
    #x = arr_data[:,1:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    ## create Gaussian model
    #nNeigbor =  121
    #weit_G = Kernel.GWR_W(coords, nNeigbor, 3)   
    #mod_G = GWGLM_Base(y, x, weit_G,0)
    
    ## 2 read non-regression data
    #flePath_pred = "E:/UK work/GWR/Code/Sample data/Georgia/Prediction/Georgia_pred.txt"
    #flds_pred = ['X', 'Y']  
    #allData_pred = FileIO.read_FILE[3](flePath_pred)
    #coords_pred = allData_pred[1]   
    
    ## change to numerical values
    #for key,val in coords_pred.items():
        #coords_pred[key] = [float(elem) for elem in val]
    
    ## 3 prediction
    #band = 121
    #wType = 3
    #mType = 0
    #myMod = pred(coords_pred, coords, band, y, x, mod_G.y_pred, wType, mType,0,None,mod_G.sigma2)
    #print myMod[0][:5]
    #print myMod[1][:5]
    #print myMod[2][:5]
    #print myMod[3][:5]
    
    ##----predicted betas----
    ##[[  1.06372455e+01  -1.51469709e-01   3.73529989e+00  -1.36821958e-01    3.21718937e-02]
    ##[  1.20290056e+01  -3.55320080e-01   3.79304225e+00  -1.18001851e-02   -9.17007929e-03]
    ##[  1.34055452e+01  -9.76930998e-02   1.03789979e+00  -3.53073753e-01    1.40077884e-01]
    ##[  1.44108143e+01  -5.90970659e-01   3.56596329e+00   8.52744829e-02   -4.82566780e-02]
    ##[  1.32801785e+01  -4.07627744e-01   3.35711847e+00  -3.22999652e-03   -2.57733572e-02]]
    ##----std. error----
    ##[[ 1.89507526  0.16442488  0.34440018  0.11167239  0.0351731 ]
    ##[ 1.8935547   0.16167172  0.34925381  0.11007352  0.03504008]
    ##[ 2.18952685  0.20330707  0.43564704  0.10849831  0.0373392 ]
    ##[ 2.35704357  0.20465527  0.40438825  0.10235797  0.03803754]
    ##[ 2.46767242  0.22147081  0.44323176  0.09397848  0.03791025]]
    ##----t stat----
    ##[[  5.61309922  -0.9212092   10.84581293  -1.22520851   0.91467314]
    ##[  6.35260525  -2.19778742  10.86041759  -0.10720276  -0.26170257]
    ##[  6.12257627  -0.48051992   2.3824328   -3.25418672   3.7514967 ]
    ##[  6.11393632  -2.88763957   8.8181674    0.8331006   -1.26865927]
    ##[  5.38166186  -1.8405484    7.57418295  -0.03436953  -0.67985186]]
    ##----local R2----
    ##[[ 0.71182756]
    ##[ 0.70123849]
    ##[ 0.42617315]
    ##[ 0.68291288]
    ##[ 0.65817466]]    
    ##--------------------end-------------------------------------
    
    ##----test prediction on non-regression points: GW Poisson model----------------
    ## Examples
    
    ## 1 read regression data
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/prediction/Tokyomortality_reg.txt"
    #flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OCC_TEC', 'OWNH', 'POP65', 'UNEMP']  
    #allData = FileIO.read_FILE[2](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    ## change to numerical values
    #for key,val in dic_data.items():
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
    
    ## create poisson model
    #band_P = 100
    #weit_P = Kernel.GWR_W(coords, band_P, 3)   
    #myMod_P = GWGLM_Base(y, x, weit_P, 1, y_off)
    
    ## 2 read non-regression data
    #flePath_pred = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/prediction/Tokyomortality_pred.txt"
    #flds_pred = ['X_CENTROID', 'Y_CENTROID']  
    #allData_pred = FileIO.read_FILE[3](flePath_pred)
    #coords_pred = allData_pred[1]   
    
    ## change to numerical values
    #for key,val in coords_pred.items():
        #coords_pred[key] = tuple(float(elem) for elem in val)
    
    ## 3 prediction
    #band = 100
    #wType = 3
    #mType = 1
    #myMod = pred(coords_pred, coords, band, y, x, myMod_P.y_pred, wType, mType,0,y_off)
    #print myMod[0][:5]
    #print myMod[1][:5]
    #print myMod[2][:5]
    #print myMod[3][:5]
    
    ##----predicted betas----
    ##[[ 0.08516783 -2.73348658 -0.34696705  2.05959735  0.08452058]
    ##[ 0.09520852 -2.75441407 -0.36152741  2.06385057  0.08442552]
    ##[ 0.04011427 -2.53307768 -0.35935116  2.08996187  0.08926424]
    ##[ 0.20521401 -2.96820255 -0.37582002  1.95344237  0.06417581]
    ##[ 0.01942714 -2.44201068 -0.38725415  2.19341452  0.09229198]]
    ##----std. error----
    ##[[ 0.11530539  0.28761105  0.09460392  0.38903842  0.0149971 ]
    ##[ 0.117262    0.29633812  0.09404849  0.39294637  0.01512106]
    ##[ 0.11180451  0.26860778  0.09342669  0.38093298  0.01506901]
    ##[ 0.12162492  0.30736433  0.09948302  0.38124388  0.01615899]
    ##[ 0.11526848  0.28627176  0.0938626   0.38980737  0.01583397]]
    ##----t stat----
    ##[[ 0.73862838 -9.50410846 -3.66757582  5.2940718   5.63579449]
    ##[ 0.81192986 -9.29483561 -3.84405346  5.25224497  5.58330591]
    ##[ 0.35878938 -9.43039569 -3.84634372  5.48642937  5.92369514]
    ##[ 1.68726941 -9.65695192 -3.77773022  5.12386546  3.97152381]
    ##[ 0.16853816 -8.53039319 -4.12575543  5.62691902  5.82873339]]
    ##----local p-dev----
    ##[[ 0.75678752]
    ##[ 0.75188292]
    ##[ 0.7655606 ]
    ##[ 0.6943642 ]
    ##[ 0.76142136]]    
    ##--------------------end-------------------------------------
    
    ##----test prediction on non-regression points: semi Poisson model----------------
    ## Examples
    
    ## 1 read regression data
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/prediction/Tokyomortality_reg.txt"
    #flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OCC_TEC', 'OWNH', 'POP65', 'UNEMP']  
    #allData = FileIO.read_FILE[2](flePath)
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
      
    ## create semi-poisson model
    #band_P = 100
    #weit_P = Kernel.GWR_W(coords, band_P, 3)   
    #myMod_P = semiGWR_Base(y, x_global, x_local, weit_P, 1, y_off)
    
    ## 2 read non-regression data
    #flePath_pred = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/prediction/Tokyomortality_pred.txt"
    #flds_pred = ['X_CENTROID', 'Y_CENTROID']  
    #allData_pred = FileIO.read_FILE[2](flePath_pred)
    #coords_pred = allData_pred[1]   
    
    ## change to numerical values
    #for key,val in coords_pred.items():
        #coords_pred[key] = tuple(float(elem) for elem in val)
    
    ## 3 prediction
    #band = 100
    #wType = 3
    #mType = 1
    #y_fix = np.dot(x_global, myMod_P.Betas_glob)
    ##print myMod_P.Betas_glob
    ##print myMod_P.std_err_glob[:5]
    ##print myMod_P.Betas_loc[:5]
    ##print myMod_P.std_err_loc[:5]
    #myMod = pred(coords_pred, coords, band, y, x_local, myMod_P.y_pred, wType, mType,0,y_off,1,y_fix, myMod_P.m_glob.FMatrix)
    #print myMod[0][:5]
    #print myMod[1][:5]
    #print myMod[2][:5]
    #print myMod[3][:5]
    
    ##----predicted betas----
    ##[[-0.27241846 -2.02866024 -0.25638356]
    ##[-0.26172831 -2.04579546 -0.27310501]
    ##[-0.27566122 -1.92079092 -0.29139884]
    ##[-0.29841654 -2.13650193 -0.15341613]
    ##[-0.26017703 -1.89828975 -0.33183925]]
    ##----std. error----
    ##[ 0.23834435  0.46941072  0.10963917]
    ##[ 0.23746876  0.47142191  0.10794635]
    ##[ 0.23508394  0.46712706  0.10424873]
    ##[ 0.23979754  0.39793233  0.13898648]
    ##[ 0.23226906  0.47027434  0.10196386]]
    ##----t stat----
    ##[[-4.37886779 -8.39420895 -2.87833782]
    ##[-4.14356782 -8.25002311 -3.0910959 ]
    ##[-4.55661942 -8.37733435 -3.28399436]
    ##[-4.66410321 -8.09794763 -1.79128399]
    ##[-4.1989381  -7.87149846 -3.71049192]]
    ##----local p-dev----
    ##[[ 0.72647443]
    ##[ 0.72172833]
    ##[ 0.73773551]
    ##[ 0.62866593]
    ##[ 0.73752188]]   
    ##--------------------end-------------------------------------
    
    ##----test local variability: basic Gaussian model--------------------------------------
    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
    #flds = ['X', 'Y', 'PctBach','PctEld', 'PctFB', 'PctPov', 'PctBlack']  
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
    #x = arr_data[:,1:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    #band = 121
    #weit = Kernel.GWR_W(coords, band, 3)  
    
    #mType = 0
    #myMod = varyTest(y, None, x, weit, 0, None, 0)
    #n = len(myMod)
    #if mType == 0:
	#print "%12s, %12s, %12s, %12s" %("f","diffp", "df","diffcri")
    #else:
	#print "%12s, %12s, %12s" % ("diffdev","diffp","diffcri")
    #for i in range(n):
	#if mType == 0:
	    #print "%12.6f, %12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2],myMod[i][3])
	#else:
	    #print "%12.6f, %12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2])	
    
    ##----output
    ##	f,        diffp,           df,      diffcri
    ##3.160098,     2.215194,   158.720116,    -2.155556
    ##10.615896,     2.510913,   158.720116,   -20.954498
    ##14.064875,     2.216439,   158.720116,   -25.845233
    ##514.201662,     1.687007,   158.720116,  -320.672548 ??
    ##8.056539,     1.650676,   158.720116,    -9.996396 
    ##----GWR4 output
    ##Variable             F                  DOF for F test  DIFF of Criterion
    ##-------------------- ------------------ ---------------- -----------------
    ##Intercept                    1.953997    2.215  158.720         0.672537
    ##PctEld                       3.555852    2.511  158.720        -3.464923
    ##PctFB                       14.039898    2.216  158.720       -25.794498
    ##PctPov                      19.408470    1.687  158.720       -28.542357
    ##PctBlack                     4.738580    1.651  158.720        -4.366333
    ##--------------------end---------------------------------------------------------------
    
    ##----test local variability: Poisson model---------------------------------------------
    ## read data
    flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt" #"E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
    flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OCC_TEC', 'OWNH', 'POP65', 'UNEMP']  
    allData = FileIO.read_FILE[2](flePath)
    dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    nobs = len(dic_data.keys()) # number of observations    
    
    # reformat data to float
    for key, val in dic_data.items():
        dic_data[key] = tuple(float(elem) for elem in val)
    
    lst_data = []
    coords = {}
    for i in range(nobs):
        coords[i] = dic_data[i][:2] # get coordinates
        lst_data.append(dic_data[i][2:])
    arr_data = np.array(lst_data)   
               
    # create x, y    
    y = np.reshape(arr_data[:,0], (-1,1))
    y_off = np.reshape(arr_data[:,1], (-1,1))
    x = arr_data[:,2:]
    x = np.hstack((np.ones(y.shape),x)) 
    
    band = 95
    weit = Kernel.GWR_W(coords, band, 3)   
    
    mType = 1
    myMod = varyTest(y, None, x, weit, 1, y_off, 0)
    n = len(myMod)
    if mType == 0:
	print "%12s, %12s, %12s, %12s" %("f","diffp", "df","diffcri")
    else:
	print "%12s, %12s, %12s" % ("diffdev","diffp","diffcri")
    for i in range(n):
	if mType == 0:
	    print "%12.6f, %12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2],myMod[i][3])
	else:
	    print "%12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2])
	    
    ##----output---------------------
    ##    diffdev,        diffp,      diffcri
    ##-87.831432,     4.464164,   -77.059868
    ##-34.224487,     4.459611,   -23.463704
    ##-5.878582,     4.258541,     4.405635
    ##-16.270581,     3.707545,    -7.296404
    ##-29.773547,     4.133795,   -19.785399
    ##----results from GWR4----------
    ##Variable             Diff of deviance    Diff of DOF    DIFF of Criterion
    ##-------------------- ------------------ ---------------- -----------------
    ##Intercept                  -87.829989         4.464163       -77.058428
    ##OCC_TEC                    -33.620726         3.136261       -26.011207
    ##OWNH                        -5.878583         4.258525         4.405596
    ##POP65                      -16.269141         3.707503        -7.295065
    ##UNEMP                      -29.772283         4.133946       -19.783775
    ##--------------------end---------------------------------------------------------------
    
    ##----test local variability: semi Poisson model---------------------------------------------
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
    
    #band = 100
    #weit = Kernel.GWR_W(coords, band, 3)   
    
    #mType = 1
    #myMod = varyTest(y, x_global, x_local, weit, 1, y_off, 0)
    #n = len(myMod)
    #if mType == 0:
	#print "%12s, %12s, %12s, %12s" %("f","diffp", "df","diffcri")
    #else:
	#print "%12s, %12s, %12s" % ("diffdev","diffp","diffcri")
    #for i in range(n):
	#if mType == 0:
	    #print "%12.6f, %12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2],myMod[i][3])
	#else:
	    #print "%12.6f, %12.6f, %12.6f" %(myMod[i][0],myMod[i][1],myMod[i][2])
    
    ##----output----------------------
    ## error message:
    ##e:\UK work\GWR\Code\lib\201312\M_GLM_semi.py:117: RuntimeWarning: overflow encountered in exp
    ##return np.exp(v+y_fix) * y_offset
    ##e:\UK work\GWR\Code\lib\201312\M_GLM_semi.py:72: RuntimeWarning: invalid value encountered in divide
      ##z = v + y_fix +(y-y_hat)/y_hat  #v + (y-y_hat)/y_hat
    ##e:\UK work\GWR\Code\lib\201312\M_GLM_semi.py:256: RuntimeWarning: invalid value encountered in absolute
      ##diff = min(abs(Betas_new-Betas))#np.sum(abs(Betas_new-Betas))/self.nVars # average difference of Betas # if using 'abs(Betas_new-Betas).all', then convergence can be very slow
    ##e:\UK work\GWR\Code\lib\201312\M_GLM_semi.py:256: RuntimeWarning: invalid value encountered in less
      ##diff = min(abs(Betas_new-Betas))#np.sum(abs(Betas_new-Betas))/self.nVars # average difference of Betas # if using 'abs(Betas_new-Betas).all', then convergence can be very slow
    ##e:\UK work\GWR\Code\lib\201312\M_GLM_semi.py:221: RuntimeWarning: invalid value encountered in greater
      ##while diff > tol and self.nIter < self.maxIter:
    ##diffdev,        diffp,      diffcri
    ##35.827679,     3.205267,    43.096005
        ##nan,          nan,          nan
    ##44.655685,     4.288239,    54.337364

    ## Results from GWR4--------------
    ##Variable             Diff of deviance    Diff of DOF    DIFF of Criterion
    ##-------------------- ------------------ ---------------- -----------------
    ##Intercept                   35.828257         3.208349        43.103482
    ##OCC_TEC                           NaN              NaN              NaN
    ##OWNH                        44.654277         4.289612        54.339002
    ##--------------------end--------------------------------------------------------------------
    
    ##-------------------------test: L2G, Gaussian model------------------------------------------
    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
    #flds = ['X', 'Y', 'PctBach','PctEld', 'PctFB', 'PctPov', 'PctBlack']  
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
    #x = arr_data[:,1:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #myMod = L2G(y, None, x, coords, 0, 3)
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #print "local Xs:"
    #print myMod[0]
    #print "global Xs:"
    #print myMod[1]
    ##----output----------------------------
    ##time: 23:38:54--23:40:16
    ##no change
    ##----GWR4 result-----------------------
    ##no change
    ##--------------------------------------end---------------------------------------------------
    
    ##-------------------------test: L2G, Poission model------------------------------------------
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
    #x = arr_data[:,2:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #myMod = L2G(y, None, x, coords, 1, 3, y_off, None, 0, 0)
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #print "local Xs:"
    #print myMod[0]
    #print "global Xs:"
    #print myMod[1]
    ##----output----------------------------
    ##time: 21:56:42--23:03:00
    ##new criteria: 360.004650697
    ##local Xs:
    ##[0, 1, 3, 4]
    ##global Xs:
    ##[2]
    ##----GWR4 result-----------------------
    ##model                                      AICc
    ##--------------------------------   ----------------
    ##GWR model before L -> G selection       365.593348 
    ##GWR model after  L -> G selection       360.004701 
    ##Improvement                             5.588647 
    ##<< Fixed (Global) coefficients >>
    ##***********************************************************
    ##Variable             Estimate        Standard Error  z(Estimate/SE)
    ##-------------------- --------------- --------------- ---------------
    ##OWNH                       -0.247146        0.077665       -3.182208
    ##--------------------------------------end---------------------------------------------------
    
    ##-------------------------test: G2L, Gaussian model------------------------------------------
    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
    #flds = ['X', 'Y', 'PctBach','PctEld', 'PctFB', 'PctPov', 'PctBlack']  
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
    #x = arr_data[:,1:]
    #x = np.hstack((np.ones(y.shape),x)) 
    
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #myMod = G2L(y, x, None, coords, 0, 3)
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    #print "local Xs:"
    #print myMod[0]
    #print "global Xs:"
    #print myMod[1]
    
    ##----output-------
    ## time: 23:54:22-23:56:10
    ## local Xs:[2, 4],global Xs:[0, 1, 3]
    ## Results from GWR4: note--if GWR4 use [2,4] as local Xs, it can get lower AICc than if use following settings.
    ##***********************************************************
    ##<< Fixed (Global) coefficients >>
    ##***********************************************************
    ##Variable             Estimate        Standard Error  t(Estimate/SE)
    ##-------------------- --------------- --------------- ---------------
    ##Intercept                  12.847997        1.428540        8.993797
    ##PctEld                     -0.080133        0.118958       -0.673621
    ##PctPov                     -0.330554        0.067310       -4.910906
    ##PctBlack                    0.088250        0.024043        3.670470