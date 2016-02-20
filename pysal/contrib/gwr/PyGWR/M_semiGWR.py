# Author: Jing Yao
# November, 2013
# Univ. of St Andrews, Scotland, UK

# For semiparametric estimaiton
from datetime import datetime
import FileIO
import Kernel
import numpy as np
import numpy.linalg as la
from M_Base import semiGWRGLM_Base
import Summary
from M_GLM import GLM_Base, GLM
#from M_OLS import OLS_Base, OLS
from M_GWGLM import GWGLM_Base
#from M_Gaussian import GWR_Gaussian_Base
#from Diagnostics import dev_mod_GLM
#m_global = {0: ,1: , 2: , 3: }
#m_local = {}

semiGWR_Names = {0: 'Semiparametric GWR: Gaussian',1: 'Semiparametric GWR: Poisson',2: 'Semiparametric GWR: Logistic'}

class semiGWR_Base(semiGWRGLM_Base):#, GWR_Base
    """
    semi-parametric GWR model: only including basic information, for bandwidth selection use. No diagnostics
    
    Method: Nakaya et al. (2005): p2705
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x_global      : array
                        n*k1, global independent variable, possibly including constant.
        x_local       : array
                        n*k2, local independent variable, possibly including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij, 
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        This term is often the size of the population at risk or the expected size of the outcome 
                        in spatial epidemiology. In cases where the offset variable box is left blank, Ni
                        becomes 1.0 for all locations.
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
        x_glob        : array
                        n*k1, global independent variable, possibly including constant.
        x_loc         : array
                        n*k2, local independent variable, possibly including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,
        mType         : integer
                        model type, 0: GLM_Gaussian, 1: GLM_Poisson, 2: GLM_Logistic, 3: basic Gaussian
        nObs          : integer
                        number of observations 
        nVars         : integer
                        number of independent variables
        nVars_glob    : integer
                        number of global independent variables
        nVars_loc     : integer
                        number of local independent variables
        m_glob        : GLM_base 
                        global model
        m_loc         : GWGLM_Base
                        local model
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
        nIter         : integer
                        number of iteration for Betas to converge
        Betas_glob    : array
                        k1, global Beta estimation
        Betas_loc     : array
                        n*k2, local Beta estimation                
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
        std_err_glob  : array
                        n*k1, standard errors of global Beta
        std_err_loc   : array
                        n*k2, standard errors of local Beta                
        t_stat_glob   : array
                        n*k1, global t-statistics
        t_stat_loc    : array
                        n*k2, local t-statistics               
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
        var_Betas     : array
                        Variance covariance matrix (kxk) of betas 
    """
    def __init__(self, y, x_global, x_local, kernel, mType=0, offset=None, sigma2_v1=False, tol=1.0e-6, maxIter=200):
            """
            Initialize class
            """
            self.x_glob = x_global
            self.x_loc = x_local
            self.y = y * 1.0           
            self.kernel = kernel
            self.mType = mType
            self.nObs = len(y) 
            self.nVars_glob = len(x_global[0])
            self.nVars_loc = len(x_local[0])
            self.nVars = self.nVars_glob + self.nVars_loc
            self.sigma2v1 = sigma2_v1
            self.tol = tol
            self.maxIter = maxIter
            if offset is None: # default offset variable
                self.offset = np.ones(shape=(self.nObs,1))   
            else:
                self.offset = offset * 1.0
            
            self.Betas_glob, self.Betas_loc, self.m_glob, self.m_loc = self._get_Betas() #self._get_Betas_book() #self._get_Betas()
            self.y_pred = self.m_glob.y_pred
            self.res = self.y - self.y_pred  
            
            self._cache = {} 
            if len(kernel.w.keys())== self.nObs:
                if sigma2_v1:
                    self.sigma2 = self.sigma2_v1
                else:
                    self.sigma2 = self.sigma2_v1v2   
    
                
    def _get_Betas(self):
        """
        Using iterative procedure to get Beta estimation
        
        Method: Tomoki et al. (2005) (41)-(44)
        
        model type: 0: GLM_Gaussian, 1: GLM_Poisson, 2: GLM_Logistic, 3: basic Gaussian
        """
        #GLM_Base
        #OLS_Base
        #GWGLM_Base
        #GWR_Gaussian_Base
        
        #Betas_loc,Betas_glob
        #1----using global model to get the inital values of yhat----------------------------------
        x_all = np.hstack((self.x_glob,self.x_loc)) # all X variables 
        #if self.mType == 3: # basic Gaussian
            #m_glob = OLS_Base(self.y, x_all)
            #dev_glob = m_glob.logll # get deviance of residuals, used for convergence decision
        #else:               # GLM
        m_glob = GLM_Base(self.y, x_all, self.mType, self.offset, self.sigma2v1, None, None, None, self.tol, self.maxIter)
        dev_glob = m_glob.dev_res # get deviance of residuals, used for convergence decision
        fMatrix = m_glob.FMatrix
        
        Betas_glob = np.zeros(shape=(self.nVars_glob, 1))
        Betas_loc = np.zeros(shape=(self.nObs,self.nVars_loc))
        #Betas_loc_ini = np.zeros(shape=(self.nVars_loc, 1))
        
        Betas_glob = m_glob.Betas[:self.nVars_glob] #(Betas_glob: k*1)
        #Betas_glob_ini = m_glob.Betas[:self.nVars_glob] 
        #Betas_loc_ini = m_glob.Betas[self.nVars_glob:]
        for i in range(self.nVars_loc):
            Betas_loc[:,i] = m_glob.Betas[self.nVars_glob+i]
        
        diff = 1e6
        nIter = 0
        
        while diff > self.tol and nIter < self.maxIter:
            nIter += 1
            #print "nIter: %d" % (nIter)
            dev_old = dev_glob
            
            #print "%s: %d" % ("nIter", nIter)
            
            #2----estimate local model------------------------------------------------------------------
            # get adjusted residuals            
            y_hat = np.dot(self.x_glob, Betas_glob)
            y_res = self.y - y_hat
            
            #if self.mType == 3: # basic Gaussian
                #m_loc = GWR_Gaussian_Base(y_res, self.x_loc,self.kernel)                
                ##dev_loc = m_loc.logll # get deviance of residuals, used for convergence decision
            #else:               # GLM
            m_loc = GWGLM_Base(self.y, self.x_loc, self.kernel, self.mType, self.offset, y_hat, fMatrix, Betas_loc, False, self.tol, self.maxIter) # Betas_loc[0]
            sMatrix = m_loc.SMatrix
                
            Betas_loc = m_loc.Betas
            y_hat = np.reshape(np.sum(self.x_loc * Betas_loc, axis=1), (-1,1))# np.sum(self.x_loc * Betas_loc) 
            y_res = self.y - y_hat
                #wx = m_loc.w
                #dev_glob_new = m_loc.dev_res # get deviance of residuals, used for convergence decision
        
            #3----estimate global model------------------------------------------------------------------
            #if self.mType == 3: # basic Gaussian
                #m_glob = OLS_Base(y_res, self.x_glob) #self._update_Betas_fixed(self.x_glob, m_loc.w, self.y-y_hat)
                #dev_glob = m_glob.logll
                ##dev_glob_new = m_loc.logll # get deviance of residuals, used for convergence decision
            #else:               # GLM
            m_glob = GLM_Base(self.y, self.x_glob, self.mType, self.offset, self.sigma2v1, y_hat, sMatrix, Betas_glob, self.tol, 5) #Betas_glob
                #y_hat = np.sum(self.x_loc * m_loc.Betas) 
            dev_glob = m_glob.dev_res
            fMatrix = m_glob.FMatrix
                #wx = m_loc.w
                #dev_glob_new = m_loc.dev_res # get deviance of residuals, used for convergence decision
            Betas_glob = m_glob.Betas
            
        
            #4----repeat (2) and (3) until convergence--------------------------------------------------
            if  self.mType == 0 and nIter == 3:#self.mType == 3 or
                diff = 0.0 # stop if Gaussian models
            else:
                diff = abs(dev_old - dev_glob)
                #print diff
                
        self.nIter = nIter
        #print m_glob.std_err
        #y_glob = np.dot(self.x_glob, Betas_glob)
        #tMatrix = m_loc.SMatrix + y_glob - np.dot(m_loc.SMatrix, y_glob)
        #print np.trace(m_glob.tMatrix) # method: Nakaya et al.(2005): (49)
        #print np.trace(tMatrix)
        #print np.trace(m_loc.SMatrix)
        #print m_loc.std_err[:5]
        #print m_loc.localpDev[:5]
                
        return Betas_glob, Betas_loc, m_glob, m_loc
    
    def _get_Betas_book(self):
        """
        from GWR book: pp67
        """
        # 1
        #xa_res = np.zeros(shape=(self.nObs,self.nVars_glob))
        dic_res = {}
        for i in range(self.nVars_glob):
            y = np.reshape(self.x_glob[:,i],(-1,1))
            m_loc = GWR_Gaussian_Base(y,self.x_loc, self.kernel)
            #y_pred = np.reshape(np.sum(self.x_loc * m_loc.Betas, axis=1), (-1,1))
            #print xa_res.shape
            #print m_loc.res.shape
            dic_res[i] = m_loc.res
            #xa_res[:,i] = m_loc.res #y - y_pred
            
        xa_res = dic_res[0]
        for i in range(1,self.nVars_glob):
            xa_res = np.hstack((xa_res,dic_res[i]))
            
        # 2,3
        m_loc = GWR_Gaussian_Base(self.y,self.x_loc, self.kernel)
        y_res = m_loc.res
        
        # 4
        m_glob = OLS_Base(y_res, xa_res)
        betas_glob = m_glob.Betas
        
        # 5
        y_res = self.y - np.dot(self.x_glob, betas_glob)
        m_loc = GWR_Gaussian_Base(y_res,self.x_loc, self.kernel)
        betas_loc = m_loc.Betas
        
        return betas_glob, betas_loc
    
     
        
class semiGWR(semiGWR_Base):
    """
    semi-parametric GWR model: only including basic information, for bandwidth selection use. No diagnostics
    
    Method: Nakaya et al. (2005): p2705
    
    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x_global      : array
                        n*k1, global independent variable, possibly including constant.
        x_local       : array
                        n*k2, local independent variable, possibly including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij, 
        mType         : integer
                        model type, 0: GLM_Gaussian, 1: GLM_Poisson, 2: GLM_Logistic, 3: basic Gaussian
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        This term is often the size of the population at risk or the expected size of the outcome 
                        in spatial epidemiology. In cases where the offset variable box is left blank, Ni
                        becomes 1.0 for all locations.
        sigma2_v1     : boolean
                        sigma squared, whether use (n-v1) as denominator
        tol:            float
                        tolerence for estimation convergence
        maxIter       : integer
                        maximum number of iteration if convergence cannot arrived to the tolerance
        y_name        : string
                        field name for y
        x_name_glob   : list of strings
                        field names of x, not include constant
        x_name_loc    : list of strings
                        field names of x, include constant
        y_off_name    : string
                        field name for offset variable
        fle_name      : string
                        data file name
                   
    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        x_glob        : array
                        n*k1, global independent variable, possibly including constant.
        x_loc         : array
                        n*k2, local independent variable, possibly including constant.
        kernel        : GWR_W object
                        weit.w: n*n, weight, {0:[w_00, w_01,...w_0n-1], 1:[w_10, w_11,...w_1n-1],...}. key: id of point (i), value: weight wij,
        mType         : integer
                        model type, 0: Gaussian, 1: Poisson, 2: Logistic
        nObs          : integer
                        number of observations 
        nVars         : integer
                        number of independent variables
        nVars_glob    : integer
                        number of global independent variables
        nVars_loc     : integer
                        number of local independent variables
        m_glob        : GLM_base 
                        global model
        m_loc         : GWGLM_Base
                        local model
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
        nIter         : integer
                        number of iteration for Betas to converge
        Betas_glob    : array
                        k1, global Beta estimation
        Betas_loc     : array
                        n*k2, local Beta estimation                
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
        std_err_glob  : array
                        n*k1, standard errors of global Beta
        std_err_loc   : array
                        n*k2, standard errors of local Beta                
        t_stat_glob   : array
                        n*k1, global t-statistics
        t_stat_loc    : array
                        n*k2, local t-statistics               
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
        var_Betas     : array
                        Variance covariance matrix (kxk) of betas 
    """
    def __init__(self, y, x_global, x_local, kernel, mType=0, offset=None, sigma2_v1=False, tol=1.0e-6, maxIter=200, y_name="", y_off_name="", x_name_glob=[], x_name_loc=[], fle_name="", summaryGLM=False):
        """
        Initialize class
        """
        semiGWR_Base.__init__(self, y, x_global, x_local, kernel, mType, offset, sigma2_v1, tol, maxIter)     
        self.y_name = y_name
        self.x_name_glob = x_name_glob
        self.x_name_loc = x_name_loc
        if y_off_name is None:
            y_off_name = " "
        self.y_off_name = y_off_name
        
        # add x names-global
        n_xname = len(self.x_name_glob)
        if n_xname < self.nVars_glob:
            for i in range(n_xname,self.nVars_glob):
                self.x_name_glob.append("name" + str(i))
        # add x names-local        
        n_xname = len(self.x_name_loc)
        if n_xname < self.nVars_loc:
            for i in range(n_xname,self.nVars_loc-1):
                self.x_name_loc.append("name" + str(i))          
        if len(self.x_name_loc) < self.nVars_loc:
            self.x_name_loc.insert(0,'Intercept')
        self.fle_name = fle_name    
        self.mName = semiGWR_Names[self.mType]
        
        if summaryGLM:
            x = np.hstack((x_global,x_local)) 
            x_name = [ elem for elem in x_name_glob]
            x_name.extend(x_name_loc)
            #if mType == 0:
                #self.OLS = OLS(y,x,sigma2_v1,y_name,x_name,fle_name)
            #else:
            self.GLM = GLM(y, x, mType, offset, sigma2_v1, tol, maxIter, y_name, y_off_name, x_name, fle_name)        
        
        Summary.semiGWR(self)

        #if mType == 0:
            #Summary.GWR(GWRMod=self)
        #else:
            #Summary.GWGLM(self)
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
        
    #**********************************2. GWR Poisson (adaptive bandwithd: bisquare)*************************
    #******************************************************************************************************
    # read data
    flePath = "E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"#"E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt" #"E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
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
    x_local = arr_data[:,2:4]
    x_local = np.hstack((np.ones(y.shape),x_local))    
    x_global = arr_data[:,4:]
    
    
    band = 100
    weit = Kernel.GWR_W(coords, band, 3)   
    
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    begin_t = datetime.now()
    #myMod = semiGWR_Base(y,x_global,x_local,weit,1, y_off,False, 1e-6, 500)
    myMod = semiGWR(y, x_global, x_local, weit, 1, y_off, False, 1.0e-6, 500, 'db2564', 'eb2564', ['POP65', 'UNEMP'], ['OCC_TEC','OWNH'], flePath, True)
    end_t = datetime.now()
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    
    #print myMod.tr_S
    #print myMod.tr_SWSTW
    #print myMod.dev_res
    #k = myMod.tr_S  
    #aic = myMod.dev_res + 2.0 * k  
    #aicc = aic + 2 * k * (k + 1.0) / (myMod.nObs - k - 1.0) 
    #bic = myMod.dev_res + k*np.log(myMod.nObs) 
    #pdev = 1 - myMod.dev_res/dev_mod_GLM(myMod)
    #print aic
    #print aicc
    #print bic
    #print pdev
    #print myMod.m_loc.std_err[:5]
    #print myMod.Betas_glob
   # print myMod.Betas_loc[:5]
    #print myMod.y_pred[:5]
    #print myMod.localpDev[:5]
    #print myMod.summary
    myMod.summary["BeginT"] = "%-21s: %s %s\n\n" % ('Program started at', datetime.date(begin_t), datetime.strftime(begin_t,"%H:%M:%S"))
    myMod.summary["EndT"] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(end_t), datetime.strftime(end_t,"%H:%M:%S"))
    print myMod.summaryPrint()
    #print myMod.std_err_loc[:5]
    
    #---------------------from GWR4 --------------------------------------
    #***********************************************************
    #<< Fixed (Global) coefficients >>
    #***********************************************************
     #Variable             Estimate        Standard Error  z(Estimate/SE)
    #-------------------- --------------- --------------- ---------------
    #POP65                       2.813380        0.431748        6.516258
    #UNEMP                       0.130027        0.039205        3.316610
    
    #----------------my output--------------------------------------------
    ##----y_pred
    #[[ 200.04189547] [ 108.5679958 ] [  74.54767083] [  53.76983759] [  72.78668243]]
    # 
    ##----m_loc.std_err[:5]
     #[[ 0.21008986  0.52936249  0.09657936]
     # [ 0.20848776  0.64923617  0.14113931]
     # [ 0.23081253  0.56380833  0.09380941]
     # [ 0.20905738  0.54908947  0.11990578]
     # [ 0.22974921  0.51464929  0.10537429]]
    ##----trace(T)
    # 17.0129975455
    ##----trace(SWS'W-1)
    # 25.3883343987
    ##----m_loc.localpDev[:5]
    #[[ 0.28213688] [ 0.09427445] [ 0.52224487] [-0.01094082] [ 0.28052743]]
    ##----Beta global
    #[[ 2.81338914] [ 0.13002816]]
    ##----Beta local
    #[[-0.50312233 -0.6581208  -0.084741  ]
     #[-0.5287654  -1.28105759  0.02769636]
     #[-0.44026769 -1.285055   -0.09193311]
     #[-0.56136661 -0.41576658 -0.01719442]
     #[-0.52588158 -0.83807734 -0.00495558]]
    ##----dev_res
    # 393.490447078
    ##----aic
    #427.516442169
    ##----aicc
    #430.028502893
    ##----bic
    #488.224678453
    ##----pdev
    #0.590217993942
    ##----summary
   
    #Summary: Geographically Weighted Regression
    #---------------------------------------------------------------------------
    
    #Program started at   : 2013-11-25 23:10:44
    #Data filename: E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt
    #Number of observations:                       262
    #Number of Variables:                          5
    
    #Model settings:
    #---------------------------------------------------------------------------
    #Model type:                                   Semiparametric GWR: Poisson
    #Geographic kernel:                            Adaptive bi-square
    
    #Modelling options:
    #---------------------------------------------------------------------------
    
    #Variable settings:
    #---------------------------------------------------------------------------
    #Dependent variable:                                          db2564
    #Offset variable:                                    eb2564
    #Independent variable with varying (Global) coefficient:       POP65
    #Independent variable with varying (Global) coefficient:       UNEMP
    #Independent variable with varying (Local) coefficient:       Intercept
    #Independent variable with varying (Local) coefficient:       OCC_TEC
    #Independent variable with varying (Local) coefficient:       OWNH
    
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
    #(Warning: trace(S) is smaller than trace(SS). It means the variance of the predictions is inadequately inflated.)
    #Note: n - trace(S) is used for computing the error variance as the degree of freedom.
    #Effective number of parameters (model: trace(S)):               17.012998
    #Effective number of parameters (variance: trace(S'WSW^-1))      25.388334
    #Degree of freedom (model: n - trace(S)):                       244.987002
    #Degree of freedom (residual: n - trace(S)):                    244.987002
    #Classic AIC:                                    427.516442
    #AICc:                                           430.028503
    #BIC:                                            488.224678
    #Null deviance:                                  960.243352
    #Residual deviance:                              393.490447
    #Percent deviance explained:                       0.590218
    
    #<< Fixed (Global) coefficients >>
    #Variable                         Estimate       Standard Error            t(Est/SE)              p-value
    #---------------------------------------------------------------------------------------------------------
    #POP65                            2.813389             0.431749             6.516261             0.000000
    #UNEMP                            0.130028             0.039205             3.316625             0.000911
    
    #<< Geographically varying (Local) coefficients >>
    #Summary statistics for varying (Local) coefficients
    
    #Variable                             Mean                  STD
    #-------------------- -------------------- --------------------
    #Intercept                       -0.448615             0.175888
    #OCC_TEC                         -1.598931             0.732904
    #OWNH                            -0.035491             0.188644
    
    #Variable                              Min                  Max                Range
    #-------------------- -------------------- -------------------- --------------------
    #Intercept                       -0.972853            -0.142328             0.830525
    #OCC_TEC                         -2.568877             0.694802             3.263678
    #OWNH                            -0.466993             0.308218             0.775211
    
    #Variable                     Lwr Quartile               Median         Upr Quartile
    #-------------------- -------------------- -------------------- --------------------
    #Intercept                       -0.527652            -0.440034            -0.326644
    #OCC_TEC                         -2.067053            -1.838921            -1.358968
    #OWNH                            -0.122391            -0.038240             0.111484
    
    #Variable                  Interquartile R           Robust STD
    #-------------------- -------------------- --------------------
    #Intercept                        0.201007             0.149005
    #OCC_TEC                          0.708085             0.524896
    #OWNH                             0.233875             0.173369
    
    #(Note: Robust STD is given by (interquartile range / 1.349) )
    
    #GWR Analysis of Deviance Table
    #---------------------------------------------------------------------------
    #Source                           Deviance                   DF          Deviance/DF 
    #-------------------- -------------------- -------------------- -------------------- 
    #Global model                   389.281580           257.000000             1.514714
    #GWR model                      393.490447           244.987002             1.606169
    #Difference                      -4.208867            12.012998            -0.350359
    
    #Program terminated at: 2013-11-25 23:10:44

    ## read data
    #flePath = "E:/UK work/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt" #"E:/Research/GWR/Code/Sample data/TokyomortalitySample/Tokyomortality.txt"
    #flds = ['X_CENTROID', 'Y_CENTROID', 'db2564', 'eb2564', 'OWNH','OCC_TEC',  'POP65', 'UNEMP']  
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
    #x_local = arr_data[:,3:]
    ##x_local = np.hstack((np.ones(y.shape),x_local))    
    #x_global = np.reshape(arr_data[:,2], (-1,1))
    #x_global = np.hstack((np.ones(y.shape),x_global))
    
    
    #band = 81
    #weit = Kernel.GWR_W(coords, band, 3)   
    
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    ##myMod = semiGWR_Base(y,x_global,x_local,weit,1, y_off,False, 1e-6, 500)
    #myMod = semiGWR(y, x_global, x_local, weit, 1, y_off, False, 1.0e-6, 200, 'db2564', 'eb2564', ['POP65', 'UNEMP'], ['OCC_TEC','OWNH'], flePath, True)
    #print datetime.strftime(datetime.now(),"%H:%M:%S")
    
    ##print myMod.tr_S
    ##print myMod.tr_SWSTW
    ##print myMod.dev_res
    ##k = myMod.tr_S  
    ##aic = myMod.dev_res + 2.0 * k  
    ##aicc = aic + 2 * k * (k + 1.0) / (myMod.nObs - k - 1.0) 
    ##bic = myMod.dev_res + k*np.log(myMod.nObs) 
    ##pdev = 1 - myMod.dev_res/dev_mod_GLM(myMod)
    ##print aic
    #print myMod.aicc
    ##print bic
    ##print pdev
    ##print myMod.m_loc.std_err[:5]
    ##print myMod.summary

     
   
    