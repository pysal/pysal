# Author: Jing Yao
# October, 2013
# Univ. of St Andrews, Scotland, UK

# For Model estimation
import FileIO
import Kernel
#import Diagnostics
import numpy as np
import numpy.linalg as la
from M_Base import Reg_Base, GWRGLM_Base
from M_GLM import GLM
#from M_OLS import OLS
from datetime import datetime
import Summary

gwglm_Names = {0: 'GWR: Gaussian',1: 'GWR: Poisson',2: 'GWR: Logistic'}


def link_G(v, y, y_offset, y_fix):
    """
    link function for Gaussian model
    
    Method: p189, Table(8.1), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        v              : array
                         n*1, v =  Beta * X 
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
    
    Return:
           ey          : array
                         n*1, E(y)
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    #y_hat = get_y_hat(0, v, y_offset)#v
    n = len(y)
    w = np.ones(shape=(n,1))
    z = y -y_fix    
    
    return z, w

def link_P(v, y, y_offset, y_fix):
    """
    link function for Poisson model
    
    Method: p189, Table(8.1), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        v              : array
                         n*1, v =  Beta * X 
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
    
    Return:
           ey          : array
                         n*1, E(y)
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    #ey = np.exp(v)
    y_hat = get_y_hat(1, v, y_offset, y_fix) #ey * y_offset
    w = y_hat
    z = v + (y-y_hat)/y_hat     
        
    return z, w

def link_L(v, y, y_offset, y_fix):
    """
    link function for Logistic model
    
    Method: p189, Table(8.1), Fotheringham, Brunsdon and Charlton (2002)
    
    Arguments:
        v              : array
                         n*1, v =  Beta * X 
        y              : array
                         n*1, dependent variable
        y_offset:      : array
                         n*1, for Poisson model
    
    Return:
           ey          : array
                         n*1, E(y)
           z           : array
                         n*1, adjusted dependent variable
           w           : array
                         n*1, weight to multiple with x
    """
    #ey = v
    y_hat = get_y_hat(2, v, y_offset, y_fix) #1.0 / (1.0 + np.exp(-v))
    deriv = y_hat * (1.0 - y_hat)
    n = len(y)
    for i in range(n):
        if (deriv[i] < 1e-10):
            deriv[i] = 1e-10
    z = v + (y - y_hat) / deriv
    w = deriv #y_hat * (1.0 - y_hat)
        
    return z, w    

def get_Betas_ini(y,x):
    """
    get initial beta using global model
    """
    xtx = np.dot(x.T, x)
    xtx_inv = la.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, x.T)
    beta = np.dot(xtx_inv_xt, y) 
    
    return beta    

def get_y_hat(mType, v, y_offset, y_fix):
    """
    get y_hat
    """
    if mType == 0:
        return v+y_fix
    if mType == 1:
        return np.exp(v+y_fix) * y_offset 
    if mType == 2:
        return 1.0/(1 + np.exp(-v-y_fix))     

#------------------Global variable------------------------
get_link = {0: link_G, 1: link_P, 2: link_L}
#---------------------------------------------------------

class GWGLM_Base(GWRGLM_Base):
    """
    Geographically weighted generalised linear model (GWGLM): Gaussian, Poisson and logistic, only including basic information, for bandwidth selection use. No diagnostics
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij, 
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        This term is often the size of the population at risk or the expected size of the outcome 
                        in spatial epidemiology. In cases where the offset variable box is left blank, Ni
                        becomes 1.0 for all locations.
        Beta_ini      : array
                        k*1, initial values of Betas
        sigma2_v1     : boolean
                        sigma squared, whether use (n-v1) as denominator
        tol:            float
                        tolerence for estimation convergence
        maxIter       : integer
                        maximum number of iteration if convergence cannot arrived to the tolerance
                   
    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        nObs          : integer
                        number of observations 
        nVars         : integer
                        number of independent variables
        ey            : array
                        n*1, E(y)
        g_ey          : array
                        n*1, g(E(y)), g() is link function, for Poisson model, g()=ln()
        y_mean        : float
                        Mean of y
        y_std         : float
                        Standard deviation of y      
        tol:          : float
                        tolerance for estimation convergence
        nIter         : list
                        number of iteration for Betas to converge for each observation
        Betas         : array
                        n*k, Beta estimation
        w             : array
                        n*1, final weight used for x
        z             : array
                        n*1, final adjusted dependent variable            
        y_pred        : array
                        n*1, predicted value of y
        res           : array
                        n*1, residuals
        sigma2        : float
                        sigma squared
        sigma2_v1     : float
                        sigma squared, use (n-v1) as denominator
        sigma2_v1v2   : float
                        sigma squared, use (n-2v1+v2) as denominator
        sigma2_ML     : float
                        sigma squared, estimated using ML          
        std_res       : array
                        n*1, standardised residuals   
        std_err       : array
                        n*k, standard errors of Beta
        t_stat        : array
                        n*k, local t-statistics
        localR2       : array
                        n*1, local R square          
        SMatrix       : array
                        n*n, S matrix (hat matrix) is used to compute diagnostics  Tomoki et al. (2005): (30) 
        tr_S          : float
                        trace of S matrix
        tr_STS        : float
                        trace of STS matrix
        CCT:          : array
                        n*k, to calculate the variance of Betas  
        CooksD        : array
                        n*1, Cook's D
        influ         : array
                        n*1, leading diagonal of S matrix    
        var_Betas     : array
                        Variance covariance matrix (kxk) of betas 
    """
    def __init__(self, y, x, kernel, mType=0, offset=None, y_fix =None, fMatrix=None, Betas_ini=None, sigma2_v1=False, tol=1.0e-6, maxIter=200):
            """
            Initialize class
            """
            self.x = x
            self.y = y * 1.0           
            self.kernel = kernel
            self.mType = mType
            self.nObs, self.nVars = x.shape  
            self.tol = tol
            self.maxIter = maxIter
            self.Betas_ini = Betas_ini
            if offset is None: # default offset variable
                self.offset = np.ones(shape=(self.nObs,1))   
            else:
                self.offset = offset * 1.0
                
            if y_fix is None:
		self.y_fix = np.zeros(shape=(self.nObs,1)) 
	    else:
		self.y_fix = y_fix
		
            ey = self.y/self.offset
            
            if self.mType == 0: # Gaussian
                g_ey = self.y 
            if self.mType == 1: # Poisson
                g_ey = np.log(ey) 
            if self.mType == 2:  # logistic
                theta = 1.0 * np.sum(self.y)/self.nObs
                id_one = self.y == 1
                id_zero = self.y == 0
                g_ey = np.ones(shape=(self.nObs,1)) 
                g_ey[id_one] = np.log(theta/(1.0-theta))
                g_ey[id_zero] = np.log((1.0 - theta)/theta)            
            
                            
            # get statistics
            #self.RMatrix = np.zeros(shape=(self.nObs,self.nObs)) # R matrix
	    nloops = len(self.kernel.w.keys())
            self.SMatrix = np.zeros(shape=(self.nObs,self.nObs)) # S matrix
            self.CCT = np.zeros(shape=(nloops,self.nVars)) # CCT to calculate variance of Betas
            
            self.Betas, self.z, self.w, self.nIter, v_final = self._get_Betas(g_ey, fMatrix)
            self.y_pred = get_y_hat(self.mType, v_final, self.offset, self.y_fix) #np.exp(v) * self.offset 
            self.res = self.y - self.y_pred   
            
            #print self.nIter
            #print self.Betas[:5]
            #print self.y_pred[:5]           
            
            self._cache = {}
            
            #print self.tr_S
            #print self.t_stat[:5]
            #print self.std_err[:5]
            #print self.Betas[:5]
            #print "ok!"
            
	    if len(kernel.w.keys())== self.nObs:
		if sigma2_v1:
		    self.sigma2 = self.sigma2_v1
		else:
		    self.sigma2 = self.sigma2_v1v2   
            
    def _get_Betas(self, g_ey, fMatrix):
        """
        get Beta estimations
        
        Methods: p189, Iteratively Reweighted Least Squares (IRLS), Fotheringham, Brunsdon and Charlton (2002)
                 Tomoki et al. (2005), (18)-(25)  
        """        
	nloops = len(self.kernel.w.keys())
        nIter = [0]*nloops       
        #lst_xtwx_inv_xtw = []
	
        Betas = np.zeros(shape=(self.nObs,self.nVars)) 
        zs = np.ones(shape=(self.nObs,1)) 
        vs = np.zeros(shape=(self.nObs,1)) 
        ws = np.zeros(shape=(self.nObs,1))
        
        # 1 get initial Betas estimation using global model
        if self.Betas_ini is None:
            Betas_ini = get_Betas_ini(g_ey-self.y_fix,self.x) #  
            Betas_old = np.zeros(shape=(self.nVars,1)) 
        else:
            Betas_ini = self.Betas_ini
            Betas_old = np.zeros(shape=(self.nObs,self.nVars)) 
        
        
        # estimate Betas for each observation
        for i in range(nloops):    # self.nObs
            #print "i: %d" % (i)
            diff = 1e6 
            #Betas_old = np.zeros(shape=(self.nVars,1)) 
            
            Betas_old = Betas_ini
            if self.Betas_ini is None:
                v = np.dot(self.x, Betas_old) 
            else:
                v = np.reshape(np.sum(self.x * Betas_old, axis=1),(-1,1)) 
            #loop
            while diff > self.tol and nIter[i] < self.maxIter:
                nIter[i] += 1 
                #print "nIter: %d" % (nIter[i])
            
                # 2 construct the adjusted dependent variable z for regression point i                
                z, w_new = get_link[self.mType](v, self.y, self.offset, self.y_fix)                 
               
                #print "y_hat:"
                #print y_hat[:5]                
                
                # 3 regress z on the xs with weights wi. Obtain a new set of parameter estimates
                #Betas_new, lst_xtwxinvxtw = self._get_Betas_IRLS(z, w_new) #self._get_Betas_IRLS(wz, wx)
                arr_w = np.reshape(np.array(self.kernel.w[i]), (-1, 1)) * w_new #w_new[i]
                w_i = np.sqrt(arr_w)
                xw = self.x * w_i            
                xtwx = np.dot(xw.T, xw)
                xtwx_inv = la.inv(xtwx)              
                xtw = (self.x * arr_w).T
                xtwx_inv_xtw = np.dot(xtwx_inv, xtw) # CMatrix[i]
                Betas_new = np.dot(xtwx_inv_xtw, z) #(np.dot(xtwx_inv_xtw, z)).T # k*1           
                
                
                # 4 determin convergence or number of iteration
                v_new = np.dot(self.x, Betas_new)  #np.reshape(np.sum(self.x * Betas_new, axis=1), (-1,1))         
                            
                
                if self.mType == 0:
                    diff = 0 # Gaussian model
                else:
                    if nIter[i] > 1:
                        diff = np.min(abs(Betas_new - Betas_old)) # minimum residual
                        #print diff
		# 5 update variables
                v = v_new
                Betas_old = Betas_new 	
		
            
            Betas[i,:] = Betas_new.T
            #print "nIter: %d" % (nIter[i])
            #print Betas[i,:] 
            zs[i] = z[i]
            vs[i] = v[i]
            ws[i] = w_new[i]
            #lst_xtwx_inv_xtw.append(xtwx_inv_xtw)
            ri = np.dot(self.x[i], xtwx_inv_xtw) #  R matrix in Tomoki (2005)--(27)
            self.SMatrix[i] = ri * np.reshape(z, (1,-1)) # get corrected S matrix:  s_ij = R_ij * zj(i) / zj(j)  
	    if fMatrix is None:
		fMatrix = np.zeros(shape=(self.nObs,self.nObs))
	    cMatrix = xtwx_inv_xtw - np.dot(xtwx_inv_xtw, fMatrix) # get corrected C matrix
            self.CCT[i] = np.reshape(np.diag(np.dot(cMatrix, cMatrix.T/w_new)), (1,-1)) # get CCT, using fMatrix                 
        
	#---------------------zs can be 0---------------------
	#-----------------------------------------------------
	
        self.SMatrix = self.SMatrix * 1.0/np.reshape(zs, (1,-1))
            
        return Betas, zs, ws, nIter, vs  #Betas, z, w_new, nIter, v, lst_xtwxinvxtw       
    
    #def _get_Betas_IRLS(self, y, ww):#y, x
        #"""
        #get Betas using IRLS
        
        #Methods: p189, Iteratively Reweighted Least Squares (IRLS), Fotheringham, Brunsdon and Charlton (2002)
        #"""
        
        #beta = np.zeros(shape=(self.nObs,self.nVars))        
        #lst_xtwx_inv_xtw = []
        
        #for i in range(self.nObs):            
            #arr_w = np.reshape(np.array(self.kernel.w[i]), (-1, 1)) * ww[i]
            #w_i = np.sqrt(arr_w)
            #xw = self.x * w_i            
            #xtwx = np.dot(xw.T, xw)
            #xtwx_inv = la.inv(xtwx)              
            #xtw = (self.x * arr_w).T
            #xtwx_inv_xtw = np.dot(xtwx_inv, xtw) # CMatrix[i]
            #beta[i] = (np.dot(xtwx_inv_xtw, y)).T #(np.dot(sel
            
            #lst_xtwx_inv_xtw.append(xtwx_inv_xtw)
                  
        
        #return beta, lst_xtwx_inv_xtw      
         
 
class GWGLM(GWGLM_Base):
    """
    Geographically weighted generalised linear model (GWGLM): Gaussian, Poisson and logistic, including estimations and diagnostics    
    
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,  
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        offset        : array
                        n*1, the offset variable at the ith location. 
                        This term is often the size of the population at risk or the expected size of the outcome 
                        in spatial epidemiology. In cases where the offset variable box is left blank, Ni
                        becomes 1.0 for all locations.
        tol:            float
                        tolerence for estimation convergence
        maxIter       : integer
                        maximum number of iteration if convergence cannot arrived to the tolerance
        sigma2_v1     : boolean
                        sigma squared, whether use (n-v1) as denominator
        y_name        : string
                        field name for y
        x_name        : list of strings
                        field names of x, not include constant
        y_off_name    : string
                        field name for offset variable
        fle_name      : string
                        data file name
            
    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij, 
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        nObs          : integer
                        number of observations 
        nVars         : integer
                        number of independent variables
        y_mean        : float
                        Mean of y
        y_std         : float
                        Standard deviation of y        
        Betas         : array
                        n*k, Beta estimation
        y_name        : string
                        field name for y
        x_name        : list of strings
                        field names of x, not including intercept
        fle_name      : string
                        name of data file
        mName         : string
                        model Name
        y_pred        : array
                        n*1, predicted value of y
        res           : array
                        n*1, residuals
        sigma2        : float
                        sigma squared
        sigma2_v1     : float
                        sigma squared, use (n-v1) as denominator
        sigma2_v1v2     : float
                        sigma squared, use (n-2v1+v2) as denominator
        sigma2_ML     : float
                        sigma squared, estimated using ML         
        std_res       : array
                        n*1, standardised residuals   
        std_err       : array
                        n*k, standard errors of Beta
        t_stat        : array
                        n*k, local t-statistics
        localR2       : array
                        n*1, local R square            
        SMatrix       : array
                        n*n, S matrix is used to compute diagnostics 
        tr_S          : float
                        trace of S matrix
        tr_STS        : float
                        trace of STS matrix
        CCT:          : array
                        n*k, to calculate the variance of Betas  
        CooksD        : array
                        n*1, Cook's D
        influ         : array
                        n*1, leading diagonal of S matrix
        logll         : float
                        Log-likelihood:
        aic           : float
                        AIC  
        aicc          : float
                        AICc
        bic           : float
                        BIC/MDL
        cv            : float
                        CV (cross validation)
        R2            : float
                        R square 
        R2_adj        : float
                        Adjusted R square
        summary       : string
                        summary information for model
    ----------
        
    """
    def __init__(self, y, x, kernel, mType=0, offset=None, sigma2_v1=False, tol=1.0e-6, maxIter=200, y_name="", y_off_name="", x_name=[], fle_name="", summaryGLM=False):
        """
        Initialize class
        """
        
        GWGLM_Base.__init__(self, y, x, kernel, mType, offset,None, None, None, sigma2_v1, tol, maxIter)     
        self.y_name = y_name
        self.x_name = x_name
        if y_off_name is None:
            y_off_name = " "
        self.y_off_name = y_off_name
	
	# add x names
        n_xname = len(self.x_name)
        if n_xname < self.nVars:
            for i in range(n_xname,self.nVars-1):
                self.x_name.append("name" + str(i))
        n_xname = len(self.x_name)
        if n_xname < self.nVars:
            self.x_name.insert(0,'Intercept') 
        self.fle_name = fle_name    
        self.mName = gwglm_Names[self.mType]
        
        if summaryGLM:
	    #if mType == 0:
                #self.OLS = OLS(y,x,sigma2_v1,y_name,x_name,fle_name)
	    #else:
	    self.GLM = GLM(y, x, mType, offset, sigma2_v1, tol, maxIter, y_name, y_off_name, x_name, fle_name)        
        
        #if mType == 0:
            #Summary.GWR(GWRMod=self)
        #else:
	Summary.GWGLM(self)  
	
    def summaryPrint(self):
	"""
	output string
	"""
	sumStr = '' 
	sumStr += self.summary['Caption']
	sumStr += self.summary['BeginT']
	sumStr += self.summary['DataSource']
	sumStr += self.summary['ModSettings']    
	sumStr += self.summary['ModOptions']
	sumStr += self.summary['VarSettings']    
	sumStr += self.summary['GlobResult']
	sumStr += self.summary['Glob_diag']    
	sumStr += self.summary['Glob_esti']
	sumStr += self.summary['GWRResult']     
	sumStr += self.summary['GWR_band']
	sumStr += self.summary['GWR_diag']  
	sumStr += self.summary['GWR_esti_glob'] 
	sumStr += self.summary['GWR_esti']
	sumStr += self.summary['GWR_anova'] 
	sumStr += self.summary['VaryTest']
	sumStr += self.summary['l2g']
	sumStr += self.summary['g2l']
	sumStr += self.summary['newMod']
	sumStr += self.summary['EndT']
	
	return sumStr
    

if __name__ == '__main__': 
    
    # Examples
    #**********************************1. GWR Poisson (adaptive bandwithd: bisquare)*************************
    # read data
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
    
    
    #**********************************1. GWR Poisson (adaptive bandwithd: bisquare)*************************
    #******************************************************************************************************
    band = 100
    weit = Kernel.GWR_W(coords, band, 3)  
    begin_t = datetime.now()
    print begin_t
    myMod = GWGLM(y, x, weit, 1, y_off, False, 1e-6,200,'db2564', 'eb2564', ['OCC_TEC', 'OWNH', 'POP65', 'UNEMP'],flePath, True)  
    end_t = datetime.now()
    print end_t
    #print myMod.Betas[:5]  
    #print myMod.std_err[:5]
    #print myMod.nObs
    #print myMod.nVars
    
    #print myMod.tr_S
    #print myMod.tr_SWSTW
    #print myMod.tr_STS
    #print myMod.y_pred[:5]
    #print myMod.kernel.w[3][3]
    #print myMod.aicc
    #print myMod.aic
    #print myMod.bic
    #print myMod.pdev
    #print myMod.dev_null
    #print myMod.dev_res
    #print myMod.localpDev[:5]
    #print myMod.t_stat[:5]
    #print myMod.tr_S
    #print myMod.tr_STS
    #print 2.0 * np.sum(myMod.y * np.log(myMod.y/myMod.y_pred))
    
    #print np.sqrt(myMod.sigma2)
    #print np.sqrt(myMod.sigma2_ML) 
    #print myMod.tr_S
    #print myMod.tr_SWSTW
    myMod.summary["BeginT"] = "%-21s: %s %s\n\n" % ('Program started at', datetime.date(begin_t), datetime.strftime(begin_t,"%H:%M:%S"))
    myMod.summary["EndT"] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(end_t), datetime.strftime(end_t,"%H:%M:%S"))
    print myMod.summaryPrint()
    
    
    ##----output-----------------------------------------------------------------------------
    
    ##----Betas
    #[[ 0.19092626 -1.54418423 -0.34008884  2.10622969 -0.01142313]
    #[ 0.10905261 -1.39758131 -0.14240077  1.59570915 -0.02437391]
    #[ 0.19601952 -2.2749161  -0.3390689   2.32711061  0.00755197]
    #[ 0.28153765 -1.25980594 -0.26846167  1.35138898 -0.04663554]
    #[ 0.23640517 -1.82111892 -0.32898964  2.00475412 -0.01175692]]
    ##----y_pred
    #[[[ 190.06917806] [  93.53236671] [  81.2077437 ] [  57.0423552 ] [  74.95401068]]
    ##----tr_S
    #25.1450916161
    ##----t statistics    
    #[[ 1.00709756 -3.12886799 -2.82737148  3.49925075 -0.33833961]
    #[ 0.44756247 -2.2075237  -0.80056503  1.98261206 -0.55352874]
    #[ 1.09720642 -4.88666413 -3.08255791  4.9927366   0.24210437]
    #[ 1.30867836 -2.37286068 -1.79719922  1.89124647 -1.19881015]
    #[ 1.34019988 -4.07667801 -2.89184862  3.97667666 -0.37859301]]
    ##----standard errors of Betas
    #[[ 0.1895807   0.49352809  0.12028446  0.60190876  0.03376232]
    #[ 0.24365897  0.63309912  0.17787534  0.80485194  0.04403369]
    #[ 0.17865328  0.4655356   0.10999595  0.46609922  0.03119303]
    #[ 0.21513128  0.53092284  0.1493778   0.71454938  0.03890152]
    #[ 0.17639546  0.4467164   0.11376448  0.50412802  0.03105424]]
    ##----local p deviation
    # [[ 0.43016636] [ 0.3302601 ] [ 0.62595703] [ 0.25783812] [ 0.46695577]]
    ##----summary
    #Summary: Geographically Weighted Regression
    #---------------------------------------------------------------------------
    
    #Program started at   : 2013-12-28 12:31:45
    #Data filename: E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt
    #Number of observations:                       262
    #Number of Variables:                          5
    
    #Model settings:
    #---------------------------------------------------------------------------
    #Model type:                                   GLM: Poisson
    #Geographic kernel:                            Adaptive bi-square
    
    #Modelling options:
    #---------------------------------------------------------------------------
    
    #Variable settings:
    #---------------------------------------------------------------------------
    #Dependent variable:                                          db2564
    #Offset variable:                                    eb2564
    #Independent variable with varying (Local) coefficient:       Intercept
    #Independent variable with varying (Local) coefficient:       OCC_TEC
    #Independent variable with varying (Local) coefficient:       OWNH
    #Independent variable with varying (Local) coefficient:       POP65
    #Independent variable with varying (Local) coefficient:       UNEMP
    
    #GWR (Geographically weighted regression) result
    #---------------------------------------------------------------------------
    #Geographic ranges
    #Coordinate                            Min                  Max                Range
    #-------------------- -------------------- -------------------- --------------------
    #X-coord                     276385.400000        408226.180000        131840.780000
    #Y-coord                     -86587.480000         33538.420000        120125.900000
    
    #GWR (Geographically weighted regression) bandwidth selection
    #---------------------------------------------------------------------------
    #Bandwidth size:                                 100.000000
    
    #Diagnostic information
    #---------------------------------------------------------------------------
    #Effective number of parameters (model: trace(S)):               25.145094
    #Effective number of parameters (variance: trace(S'WSW^-1))      17.142371
    #Degree of freedom (model: n - trace(S)):                       236.854906
    #Degree of freedom (residual: n - 2trace(S) + trace(S'WSW^-1)):   228.852183
    #Classic AIC:                                    361.535488
    #AICc:                                           367.110279
    #BIC:                                            451.261846
    #Null deviance:                                  960.243352
    #Residual deviance:                              311.245301
    #Percent deviance explained:                       0.675868
    
    #<< Geographically varying (Local) coefficients >>
    #Summary statistics for varying (Local) coefficients
    
    #Variable                             Mean                  STD
    #-------------------- -------------------- --------------------
    #Intercept                        0.038565             0.295395
    #OCC_TEC                         -2.132644             1.004125
    #OWNH                            -0.275802             0.157200
    #POP65                            2.169549             0.626589
    #UNEMP                            0.047531             0.038121
    
    #Variable                              Min                  Max                Range
    #-------------------- -------------------- -------------------- --------------------
    #Intercept                       -0.879764             0.408928             1.288692
    #OCC_TEC                         -3.607038             1.218879             4.825918
    #OWNH                            -0.547011             0.111386             0.658397
    #POP65                            1.319626             4.095840             2.776214
    #UNEMP                           -0.051157             0.159427             0.210584
    
    #Variable                     Lwr Quartile               Median         Upr Quartile
    #-------------------- -------------------- -------------------- --------------------
    #Intercept                        0.002327             0.090003             0.254468
    #OCC_TEC                         -2.660980            -2.503268            -1.839615
    #OWNH                            -0.375564            -0.321084            -0.208452
    #POP65                            1.677317             2.083871             2.418957
    #UNEMP                            0.022383             0.044555             0.075304
    
    #Variable                  Interquartile R           Robust STD
    #-------------------- -------------------- --------------------
    #Intercept                        0.252140             0.186909
    #OCC_TEC                          0.821365             0.608870
    #OWNH                             0.167112             0.123878
    #POP65                            0.741640             0.549770
    #UNEMP                            0.052921             0.039229
    
    #(Note: Robust STD is given by (interquartile range / 1.349) )
    
    #GWR Analysis of Deviance Table
    #---------------------------------------------------------------------------
    #Source                           Deviance                   DF          Deviance/DF 
    #-------------------- -------------------- -------------------- -------------------- 
    #Global model                   389.281580           257.000000             1.514714
    #GWR model                      311.245301           228.852183             1.360028
    #Difference                      78.036279            28.147817             2.772374
    
    #Program terminated at: 2013-12-28 12:31:45



    
    ##-----end of output---------------------------------------------------------------------
    
    #nNeigbor =  121 #145 129
    #weit = Kernel.GWR_W(coords, nNeigbor, 3, 1)   
    #myMod = GWGLM(y, x, weit, 0, None, False, 1e-6, 50, 'PctBach', ' ', ['PctEld', 'PctFB', 'PctPov', 'PctBlack'],flePath,False)
    ##GWGLM(y, x, weit, 1, y_off, False, 1e-6,50,'db2564', 'eb2564', ['OCC_TEC', 'OWNH', 'POP65', 'UNEMP'],flePath, True)  
    #print myMod.y_pred[:5]
    #print myMod.summary
    
    
   
