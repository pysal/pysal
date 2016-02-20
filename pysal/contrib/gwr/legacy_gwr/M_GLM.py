# Author: Jing Yao
# Aug, 2013
# Univ. of St Andrews, Scotland, UK

# For Model estimation: semi-parametric GWR
import FileIO
import Kernel
#import Diagnostics
import numpy as np
import numpy.linalg as la
from M_Base import Reg_Base, GWR_Base
import Summary
from datetime import datetime
#from scipy import stats

glm_Names = {0: 'Gaussian',1: 'Poisson',2: 'Logistic'}


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
    #ey = v
    n = len(y)
    w = np.ones(shape=(n,1))
    z = y #-y_fix    
    
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
    #ey = np.exp(v+y_fix)
    y_hat = get_y_hat(1, v, y_offset, y_fix)   
    w = y_hat #ey
    z = v + y_fix +(y-y_hat)/y_hat  #v + (y-y_hat)/y_hat     
    
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
    y_hat = get_y_hat(2, v, y_offset, y_fix)#1.0 / (1.0 + np.exp(-v-y_fix))
    deriv = y_hat * (1.0 - y_hat)
    n = len(y)
    for i in range(n):
        if (deriv[i] < 1e-10):
            deriv[i] = 1e-10
    z = v + y_fix + (y - y_hat) / deriv
    w = deriv #y_hat * (1.0 - y_hat)
    
    return z, w

def get_y_hat(mType, v, y_offset, y_fix):
    """
    get y_hat
    """
    if mType ==0:
        return v+y_fix
    if mType == 1:
        return np.exp(v+y_fix) * y_offset 
    if mType == 2:
        return 1.0/(1 + np.exp(-v-y_fix))  
    
#------------------Global variable------------------------
get_link = {0: link_G, 1: link_P, 2: link_L}
#---------------------------------------------------------

class GLM_Base(Reg_Base):
    """
    Generalised linear model (GLM): Gaussian, Poisson and logistic, only including basic information. No diagnostics
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant. 
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        This term is often the size of the population at risk or the expected size of the outcome 
                        in spatial epidemiology. In cases where the offset variable box is left blank, Ni
                        becomes 1.0 for all locations.        
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
        tol:          : float
                        tolerance for estimation convergence
        nIter         : integer m
                        number of iteration for Betas to converge
        Betas         : array
                        n*k, Beta estimation
        w             : array
                        n*1, final weight used for x
        y_pred        : array
                        n*1, predicted value of y
        res           : array
                        n*1, residuals         
        var_Betas     : array
                        Variance covariance matrix (kxk) of betas    
    """
    def __init__(self, y, x, mType=0, offset=None, sigma2_v1=False, y_fix =None, sMatrix=None, Betas_ini=None, tol=1.0e-6, maxIter=200):
            """
            Initialize class
            """
            self.x = x
            self.y = y * 1.0           
            self.mType = mType
            self.nObs, self.nVars = x.shape  
            self.tol = tol
            self.maxIter = maxIter            
            if offset is None: # default offset variable
                self.offset = np.ones(shape=(self.nObs,1))   
            else:
                self.offset = offset * 1.0
		
	    if y_fix is None:
		self.y_fix = np.zeros(shape=(self.nObs,1)) 
	    else:
		self.y_fix = y_fix
		
            ey = self.y/self.offset
            if mType == 0: # Gaussian
                g_ey = self.y 
            if mType == 1: # Poisson
                g_ey = np.log(ey) #ln(Y/Y_offset) = Beta*X, g()=ln()
            if mType == 2:  # logistic
                theta = 1.0 * np.sum(self.y)/self.nObs
                id_one = self.y == 1
                id_zero = self.y == 0
                g_ey = np.ones(shape=(self.nObs,1)) 
                g_ey[id_one] = np.log(theta/(1.0-theta))
                g_ey[id_zero] = np.log((1.0 - theta)/theta)
            
            diff = 1.0e6
            self.nIter = 0
            		
	    if Betas_ini is None:
		Betas = self._get_Betas_IRLS(g_ey-self.y_fix, self.x) # self._get_Betas_IRLS(g_ey-self.y_fix, self.x)
	    else:
		Betas = Betas_ini
		
            v = np.dot(self.x, Betas)
            #loop
            while diff > tol and self.nIter < self.maxIter:
                self.nIter += 1 
                #print "nIter: %d" % (self.nIter)
                #print Betas[:5]
                
                # 2 construct the adjusted dependent variable z
                z, w_new = get_link[self.mType](v, self.y, self.offset, self.y_fix)
                #ey = np.exp(v)   
                #y_hat = ey * self.offset
                #z = v + (self.ey-ey)/ey  # v + (self.y-y_hat)/y_hat #z = v + (self.ey-ey)/ey
                #print "y_hat:"
                #print y_hat[:5]
                
                # 3 define new weighted x
		ww = np.sqrt(w_new)
                wx = self.x * ww #np.array(self.kernel.w.values()) * ey 
		if sMatrix is None:
		    wz = z * ww
		    Betas_new = self._get_Betas_IRLS(wz, wx)
		else:
		    wz = (z - np.dot(sMatrix,z)) * ww
		    wx2 = (self.x - np.dot(sMatrix,self.x)) * ww
		    xtx = np.dot(wx.T, wx2)
		    xtxinv = la.inv(xtx)
		    xtz = np.dot(wx.T, wz)
                
		    # 4 regress z on the xs with weights wi. Obtain a new set of parameter estimates
		    Betas_new = np.dot(xtxinv, xtz)#self._get_Betas_IRLS(wz, wx)
                
                # 5 determin convergence or number of iteration
                v_new = np.dot(self.x, Betas_new)
                #y_hat_new = np.exp(v_new) * self.offset
		if self.mType == 0:
		    diff = 0.0
		else:
		    diff = min(abs(Betas_new-Betas))#np.sum(abs(Betas_new-Betas))/self.nVars # average difference of Betas # if using 'abs(Betas_new-Betas).all', then convergence can be very slow
                
                # 6 update variables
                v = v_new
                Betas = Betas_new
                
            self.nIter += 1
            self.w = w_new
            self.Betas = Betas_new
	    self.y_pred = get_y_hat(self.mType, v_new, self.offset, self.y_fix)
            #if self.mType ==0:
                #self.y_pred = v_new
            #if self.mType == 1:
                #self.y_pred = np.exp(v_new) * self.offset 
            #if self.mType == 2:
                #self.y_pred = 1.0/(1 + np.exp(-v)) 
            self.res = self.y - self.y_pred  
	    #res2 = np.sum(self.res**2) # sum of residual squares
	    self.FMatrix = np.zeros(shape=(self.nObs,self.nObs)) # initialize f matrix
	    
	    self._cache = {}
	    
	    # get std error of betas and global hat matrix f, and semiGWR hat matrix = S + (I-S) f = S + (I-S) x inv(xt w (I-S)x) xt w (I-S)
	    if not sMatrix is None:
		wx3 = self.x * self.w
		xws = wx3.T - np.dot(wx3.T, sMatrix)
		cMatrix = np.dot(xtxinv, xws)
		#self._cache['var_Betas'] = np.reshape(np.diag(np.dot(cMatrix,cMatrix.T/self.w)), (-1,1))
		#self.std_err = np.sqrt(varBetas)
		fMatrix = np.dot(self.x, cMatrix)
		tMatrix = sMatrix + fMatrix - np.dot(sMatrix, fMatrix)  # method: Nakaya et al.(2005): (49)
		self.tr_S = np.trace(tMatrix)
		self.FMatrix = fMatrix # global hat matrix
		
		# Gaussian model, influence			
		if self.mType == 0: 
		    self.tr_STS = np.trace(np.dot(tMatrix, tMatrix.T))
		    self.influ = np.reshape(np.diag(tMatrix),(-1,1))
		    if self.tr_S > self.tr_STS:
			self.sigma2 = self.res2/ (self.nObs - 2.0* self.tr_S + self.tr_STS)
		    else:
			self.sigma2 = self.res2/ (self.nObs - 2.0* self.tr_S)		    
		    self._cache['var_Betas'] =self.sigma2 * np.reshape( np.diag(np.dot(cMatrix,cMatrix.T/self.w)), (-1,1))
		else:
		    # tr_SWSTW
		    stw = (tMatrix * np.reshape(self.w, (-1,1))).T
		    stws = np.dot(stw, tMatrix)
		    stwsw = stws.T *1.0/self.w
		    self.tr_SWSTW = np.trace(stwsw)
		    self._cache['var_Betas'] = np.reshape(np.diag(np.dot(cMatrix,cMatrix.T/self.w)), (-1,1))
		    #if self.tr_S > self.tr_STS:
			#self.scales = np.sqrt(res2/ self.nObs - 2.0* self.tr_S + self.tr_STS)
		    #else:
			#self.scales = np.sqrt(res2/ self.nObs - 2.0* self.tr_S)		    
                                
            if sigma2_v1:
		self.sigma2 = self.sigma2_n
            else:
                self.sigma2 = self.sigma2_nk
                
            #print self.nIter
            #print self.y_pred[:5]
            #print self.Betas[:5]
            #print np.reshape(np.sqrt(np.diag(self.var_Betas)),(-1,1)) # SE of Betas
            #tStat= self.Betas/np.reshape(np.sqrt(np.diag(self.var_Betas)),(-1,1)) # t statistics
            #print tStat
            #for i in range(self.nVars):
                #print stats.norm.sf(abs(tStat[i]))*2
            #print self.sigma2_nk
            #print self.sigma2_n
            
            #xw = self.x * np.sqrt(self.w)
            #xw_inv = np.dot(xw, self.xtwxi)
            #HMatrix = np.dot(xw_inv, xw.T)          
            #dev = 2.0 * np.sum(self.y * np.log(self.y/self.y_pred))
            #ybar = np.sum(self.y)/np.sum(self.offset)
            #dev_full = 2.0 * np.sum(self.y * np.log(self.y/(ybar*self.offset)))        
            #aic = dev + 2.0*self.nVars #-2L+2K
            #k= self.nVars
            #n= self.nObs
            #aicc = aic + 2.0*k*(k+1)/(n-k-1)  
            #bic = dev + k*np.log(n)
            #print dev_full
            #print bic
            #print "ok!"
                
    
    def _get_Betas_IRLS(self, y, x):
        """
        get Betas using IRLS
        
        Methods: p189, Iteratively Reweighted Least Squares (IRLS), Fotheringham, Brunsdon and Charlton (2002)
        """      
       
        self.xtx = np.dot(x.T, x)
        xtx_inv = la.inv(self.xtx)
	self.xtxi = xtx_inv
        xtx_inv_xt = np.dot(xtx_inv, x.T)
        beta = np.dot(xtx_inv_xt, y)             
        
        return beta   
    
    #@property
    #def res2(self):
        #if 'res2' not in self._cache:
            #self._cache['res2'] = np.sum(self.res**2)
        #return self._cache['res2']
    
    @property
    def sigma2_n(self):
        if 'sigma2_n' not in self._cache:
            self._cache['sigma2_n'] = np.sum(self.w*self.res**2) / self.nObs
        return self._cache['sigma2_n']
    
    @property
    def sigma2_nk(self):
        if 'sigma2_nk' not in self._cache:
            self._cache['sigma2_nk'] = np.sum(self.w*self.res**2)/ (self.nObs-self.nVars)
        return self._cache['sigma2_nk']
    
    @property
    def var_Betas(self):
        if 'var_Betas' not in self._cache:
	    if self.mType == 0: # Gaussian		
		self._cache['var_Betas'] = np.dot(self.sigma2, self.xtxi).diagonal()
	    else:
		xtw = (self.x * self.w).T 
		xtwx = np.dot(xtw, self.x)          
		self._cache['var_Betas'] = la.inv(xtwx).diagonal() #self.xtwxi # #xtwx_inv
        return self._cache['var_Betas']  
    
    @property
    def std_err(self):  
        """
        standard errors of Betas
        
        Methods: Nakaya et al. (2005): p2703, (47)
        """
        if 'std_err' not in self._cache:
            self._cache['std_err'] = np.reshape(np.sqrt(self.var_Betas),(-1,1))
        return self._cache['std_err'] 
    
    @property
    def dev_res(self):
        """
        deviance of residuals
        """
	if 'dev_res' not in self._cache:
	    dev = 0.0
	    if self.mType == 0:
		#res2 = np.sum(self.res**2)
		dev = self.nObs * (np.log(self.res2 * 2.0 * np.pi / self.nObs) + 1.0) # -2loglikelihood
	    if self.mType == 1:
		id0 = self.y==0
		id1 = self.y<>0
	
		if np.sum(id1) == self.nObs:
		    dev = 2.0 * np.sum(self.y * np.log(self.y/self.y_pred))
		else:
		    dev = 2.0 * (np.sum(self.y[id1] * np.log(self.y[id1]/self.y_pred[id1]))-np.sum(self.y[id0]-self.y_pred[id0]))   
	    
	    if self.mType == 2:
		for i in range(self.nObs):
		    if self.y[i] == 0:
			dev += -2.0 * np.log(1.0 - self.y_pred[i])
		    else: 
			dev += -2.0 * np.log(self.y_pred[i])		
		#v = np.dot(self.x, self.Betas)
		#for i in range(self.nObs):
		    #if ((1.0 - self.y_pred[i]) < 1e-10): 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1e-10) )
		    #else: 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1.0 - self.y_pred[i]) )   
	    self._cache['dev_res'] = dev #dev[0]
	return self._cache['dev_res']
	
    
    #@property
    #def HMatrix(self):
        #if 'HMatrix' not in self._cache:
            #xw = self.x * np.sqrt(self.w)
            #xw_inv = np.dot(xw, self.xtwxi)
            #self._cache['HMatrix'] = np.dot(xw_inv, xw.T) # = W1/2X(X'WX)-1XW1/2
        #return self._cache['HMatrix']  
        
    
    #def resetBetas(self):
        #"""
        #recalculate Beta if wii changes, keep other statistics unchanges
        #to calculate CV
        #"""
        #xt = self.x.T
        #beta = np.zeros(shape=(self.nObs,self.nVars))
        #for i in range(self.nObs):
            #arr_w = np.array(self.kernel.w[i])
            #w_i = np.diag(arr_w)
            #xtw = np.dot(xt, w_i)
            #xtwx = np.dot(xtw, self.x)
            #xtwx_inv = la.inv(xtwx)
            #xtwx_inv_xt = np.dot(xtwx_inv, xt)
            #xtwx_inv_xt_w = np.dot(xtwx_inv_xt, w_i) # CMatrix[i]             
            #beta[i] = (np.dot(xtwx_inv_xt_w, self.logY)).T  
        
        #return beta            
 
class GLM(GLM_Base):
    """
    Generalised linear model (GLM): Gaussian, Poisson and logistic, including estimations and diagnostics    
    
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant. 
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
        y_name        : string
                        field name for y
        x_name        : list of strings
                        field names of x, not include constant
        y_off_name    : string
                        field name for offset variable
        
            
    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
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
        w             : array
                        n*1, final weight used for x
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
        var_Betas     : array
                        Variance covariance matrix (kxk) of betas
        std_err       : array
                        1xk array of standard errors of the betas 
        t_stat        : list of tuples
                        t statistic; each tuple contains the pair (statistic,
                        p-value), where each is a float        
        aic           : float
                        AIC  
        aicc          : float
                        AICc
        bic           : float
                        BIC/MDL       
        dev_res       : float
                        residual deviance
        dev_null      : float
                        null deviance
        pdev          : float
                        percent of deviance explained
        summary       : string
                        summary information for model
    ----------
        
    """
    def __init__(self, y, x, mType=0, offset=None, sigma2_v1=False, tol=1.0e-6, maxIter=200, y_name="", y_off_name="", x_name=[], fle_name=""):
        """
        Initialize class
        """
        
        GLM_Base.__init__(self, y, x, mType, offset, sigma2_v1, None, None,None, tol, maxIter)     
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
        self.mName = glm_Names[self.mType]
        
	#if mType == 0:
	    #Summary.OLS(OLSMod=self)
	#else:
	Summary.GLM(GLMMod=self)    
	
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
	sumStr += self.summary['GLMResult']
	sumStr += self.summary['GLM_diag']    
	sumStr += self.summary['GLM_esti']
	sumStr += self.summary['EndT']
	
	return sumStr

if __name__ == '__main__': 
    
    # Examples  
    #**********************************1. GLM Gaussian *****************************************************
    #******************************************************************************************************
    # read data
    flePath = "E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf" #"E:/Research/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf"
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
    
    begin_t = datetime.now()
    myMod = GLM(y, x, 0, None, False, 1e-6, 200, 'PctBach', ' ', ['PctEld', 'PctFB', 'PctPov', 'PctBlack'],flePath)
    end_t = datetime.now()
    
    #print myMod.aic # 969.82303767, GWR4:  971.823037
    #print myMod.aicc # 970.180180527, GWR4: 972.326031
    #print myMod.bic
    #print myMod.cv
    #print myMod.summary
    myMod.summary["BeginT"] = "%-21s: %s %s\n\n" % ('Program started at', datetime.date(begin_t), datetime.strftime(begin_t,"%H:%M:%S"))
    myMod.summary["EndT"] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(end_t), datetime.strftime(end_t,"%H:%M:%S"))
    #print myMod.summaryPrint()
    
    ##------------output---------------------
    #Summary: Ordinary Least Squares Estimation
    #---------------------------------------------------------------------------
    
    #Program started at   : 2013-12-31 12:33:04
    #Data filename: E:/UK work/GWR/Code/Sample data/Georgia/GeorgiaEduc.dbf
    #Number of observations:                       174
    #Number of Variables:                          5
    
    #Model settings:
    #---------------------------------------------------------------------------
    #Model type:                                   Gaussian
    
    #---------------------------------------------------------------------------
    #Dependent variable:                                PctBach
    
    #Global regression result
    #< Diagnostic information >
    #---------------------------------------------------------------------------
    #Residual sum of squares:                       2533.615435
    #ML based global sigma estimate:                   3.815889
    #Unbiased global sigma estimate:                   3.871926
    #-2Log-likelihood:                               959.823038
    #Classic AIC:                                    969.823038
    #AICc:                                           970.180181
    #BIC/MDL:                                        985.618314
    #CV:                                              14.561008
    #R square:                                         0.525104
    #Adjusted R square:                                0.513864
    
    #Variable                         Estimate       Standard Error            t(Est/SE)              p-value
    #---------------------------------------------------------------------------------------------------------
    #Intercept                       12.789636             1.520720             8.410249             0.000000
    #PctEld                          -0.116422             0.129842            -0.896639             0.371187
    #PctFB                            2.538762             0.284333             8.928844             0.000000
    #PctPov                          -0.272978             0.073316            -3.723328             0.000268
    #PctBlack                         0.073405             0.026214             2.800241             0.005701
    
    #Program terminated at: 2013-12-31 12:33:04
    
    ##**********************************1. GLM Poisson *****************************************************
    ##******************************************************************************************************
    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
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
    
    #myMod = GLM(y, x, 1, y_off, False, 1e-6,50,'db2564', 'eb2564', ['OCC_TEC', 'OWNH', 'POP65', 'UNEMP'],flePath)  
    
    #print 2.0 * np.sum(myMod.y * np.log(myMod.y/myMod.y_pred))  
    #print myMod.summary
    
    ##----output-----------------------------------------------------------------------------
    
    #Summary: Generalised linear model (GLM)
    #---------------------------------------------------------------------------
    
    #Program started at   : 2013-08-20 14:38:49
    #Data filename: E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt
    #Number of observations:                       262
    #Number of Variables:                          5
    
    #Model settings:
    #---------------------------------------------------------------------------
    #Model type:                                   GLM: Poisson
    
    #---------------------------------------------------------------------------
    #Dependent variable:                                 db2564
    #Offset variable:                                    eb2564
    
    #Global regression result
    #< Diagnostic information >
    #---------------------------------------------------------------------------
    #Classic AIC:                                    399.281580
    #AICc:                                           399.515955
    #BIC/MDL:                                        417.123303
    #Null deviance:                                  960.243352
    #Residual deviance:                              389.281580
    #Percent deviance explained:                       0.594601
    
    #Variable                         Estimate       Standard Error            z(Est/SE)              p-value
    #---------------------------------------------------------------------------------------------------------
    #Intercept                        0.007470             0.065139             0.114679             0.908699
    #OCC_TEC                         -2.287906             0.162000           -14.122903             0.000000
    #OWNH                            -0.259692             0.047050            -5.519470             0.000000
    #POP65                            2.199387             0.198270            11.092878             0.000000
    #UNEMP                            0.064025             0.010997             5.822059             0.000000
    
    #Program terminated at: 2013-08-20 14:38:49
    
    ##-----end of output---------------------------------------------------------------------
    
    
    ##**********************************2. GLM Logistic ****************************************************
    ##******************************************************************************************************
    ##flePath = " E:/Research/GWR/Code/Sample data/for global logistic/gorilla.csv" # http://ww2.coastal.edu/kingw/statistics/R-tutorials/text/gorilla.csv
    #flePath = "E:/Research/GWR/Code/Sample data/for global logistic/binary.csv" # http://www.ats.ucla.edu/stat/data/binary.csv
    #flds = ['admit', 'gre', 'gpa', 'rank']  
    #allData = FileIO.read_FILE[0](flePath)
    #dic_data = FileIO.get_subset(allData[0], allData[1], flds)    
    #nobs = len(dic_data.keys()) # number of observations    
    
    ## reformat data to float
    #for key, val in dic_data.items():
        #dic_data[key] = tuple(float(elem) for elem in val)
    
    #lst_data = []
    ##coords = {}
    #for i in range(nobs):
        ##coords[i] = dic_data[i][:2] # get coordinates
        #lst_data.append(dic_data[i])
    #arr_data = np.array(lst_data)   
               
    ## create x, y    
    #y = np.reshape(arr_data[:,0], (-1,1))
    ##y_off = np.reshape(arr_data[:,1], (-1,1))
    #x = arr_data[:,1:]
    #x = np.hstack((np.ones(y.shape),x))     
   
    #myMod = GLM(y, x, 2, None,1e-6,50, 'admit',None,['gre', 'gpa', 'rank'],flePath)  
    
    #print myMod.summary
    
    ##----output-----------------------------------------------------------------------------
    #Summary: Generalised linear model (GLM)
    #---------------------------------------------------------------------------
    
    #Program started at   : 2013-08-27 16:31:09
    #Data filename: E:/Research/GWR/Code/Sample data/for global logistic/binary.csv
    #Number of observations:                       400
    #Number of Variables:                          4
    
    #Model settings:
    #---------------------------------------------------------------------------
    #Model type:                                   GLM: Logistic
    
    #---------------------------------------------------------------------------
    #Dependent variable:                                  admit
    
    #Global regression result
    #< Diagnostic information >
    #---------------------------------------------------------------------------
    #Classic AIC:                                    467.441765
    #AICc:                                           467.543031
    #BIC/MDL:                                        483.407623
    #Null deviance:                                  499.976518
    #Residual deviance:                              459.441765
    #Percent deviance explained:                       0.081073
    
    #Variable                         Estimate       Standard Error            z(Est/SE)              p-value
    #---------------------------------------------------------------------------------------------------------
    #Intercept                       -3.449548             1.132846            -3.045028             0.002482
    #gre                              0.002294             0.001092             2.101005             0.036272
    #gpa                              0.777014             0.327484             2.372677             0.018137
    #rank                            -0.560031             0.127137            -4.404944             0.000014
    
    #Program terminated at: 2013-08-27 16:31:09    
    
    ##-----end of output---------------------------------------------------------------------
    
    

    