# Author: Jing Yao
# July, 2013
# Univ. of St Andrews, Scotland, UK

# For common properties of all GWR models

import numpy as np
import numpy.linalg as la

class Reg_Base(object):
    """
    Basic class including common properties for all regression models.

    Parameters
    ----------

    Attributes
    ----------
    y_mean  : float
              Mean of y
    y_std   : float
              Standard deviation of y
              
    """

    @property
    def y_mean(self):
        if 'y_mean' not in self._cache:
            self._cache['y_mean']=np.mean(self.y)
        return self._cache['y_mean']
    @property
    def y_std(self):
        if 'y_std' not in self._cache:
            self._cache['y_std']=np.std(self.y, ddof=1)
        return self._cache['y_std']
    
    @property
    def res2(self):  
        """
        sum of squared residuals
        """
        if 'res2' not in self._cache:
            self._cache['res2'] = np.sum(self.res**2)
        return self._cache['res2']   
    
    
class GWR_Base(Reg_Base):
    """
    Basic class including common properties for all GWR regression models
    
    Parameters
    ----------

    Attributes
    ----------
    res2      : float
                sum of squared residuals
    sigma2_v1 : float
                sigma squared, use (n-v1) as denominator
    sigma2_v1v2 : float
                sigma squared, use (n-2v1+v2) as denominator
    sigma2_ML : float
                sigma squared, estimated using ML          
    std_res   : array
                n*1, standardised residuals   
    std_err   : array
                n*k, standard errors of Beta
    t_stat    : array
                n*k, local t-statistics
    localR2   : array
                n*1, local R square            
    tr_S      : float
                trace of S matrix
    tr_STS    : float
                trace of STS matrix
    CooksD    : array
                n*1, Cook's D
    influ     : array
                n*1, leading diagonal of S matrix
    """
      

    @property
    def tr_S(self):  
        """
        trace of S matrix
        """
        if 'tr_S' not in self._cache:
            self._cache['tr_S'] = np.trace(self.SMatrix)
        return self._cache['tr_S']   
    
    @property
    def tr_STS(self):  
	"""
	trace of STS matrix
	"""
	if 'tr_STS' not in self._cache:
	    self._cache['tr_STS'] = np.trace(np.dot(self.SMatrix.T,self.SMatrix))
	return self._cache['tr_STS']     
        
    @property
    def y_bar(self):
        """
        weighted y mean
        """
        if 'y_bar' not in self._cache:
            arr_ybar = np.zeros(shape=(self.nObs,1))            
            for i in range(self.nObs):
                w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                sum_yw = np.sum(self.y * w_i)
                arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)    # weighted average              
            self._cache['y_bar'] = arr_ybar
        return self._cache['y_bar']          
        
    #@property
    def TSS(self):
        """
        geographically weighted total sum of squares
        
        Methods: p215, (9.9), Fotheringham, Brunsdon and Charlton (2002)
        """
        #if 'TSS' not in self._cache:
	arr_R = np.zeros(shape=(self.nObs,1))
	for i in range(self.nObs):
	    arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1)) * (self.y - self.y_bar[i])**2)   #self.y_bar[i]               
            #self._cache['TSS'] = arr_R
        return arr_R #self._cache['TSS']  
    
    #@property
    def RSS(self):
        """
        geographically weighted residual sum of squares
        
        Methods: p215, (9.10), Fotheringham, Brunsdon and Charlton (2002)
        """
        #if 'RSS' not in self._cache:
	arr_R = np.zeros(shape=(self.nObs,1))
	for i in range(self.nObs):
	    arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1)) * self.res**2) 
	#self._cache['RSS'] = arr_R
        return arr_R #self._cache['RSS']      
    
    @property
    def localR2(self):
        """
        local R square
        
        Methods: p215, (9.8), Fotheringham, Brunsdon and Charlton (2002)
        """
        if 'localR2' not in self._cache:
            self._cache['localR2'] = (self.TSS() - self.RSS())/self.TSS() #(self.TSS - self.RSS)/self.TSS
        return self._cache['localR2']    
    
    @property
    def sigma2_v1(self):  
        """
        residual variance
        
        Methods: p214, (9.6), Fotheringham, Brunsdon and Charlton (2002), only use v1
        
        """
        if 'sigma2_v1' not in self._cache:
            self._cache['sigma2_v1'] = self.res2/(self.nObs-self.tr_S) # method only use v1
            
        return self._cache['sigma2_v1']  
    
    @property
    def sigma2_v1v2(self):  
        """
        residual variance
        
        Methods: p55 (2.16)-(2.18), Fotheringham, Brunsdon and Charlton (2002), use v1 and v2 #used in GWR4
        
        """
        if 'sigma2_v1v2' not in self._cache:            
            self._cache['sigma2_v1v2'] = self.res2/(self.nObs - 2.0*self.tr_S + self.tr_STS) # method used in GWR4
            
        return self._cache['sigma2_v1v2']      
    
    @property
    def sigma2_ML(self):  
        """
        residual variance
        
        Methods: maximum likelihood
        
        """
        if 'sigma2_ML' not in self._cache:
            
            self._cache['sigma2_ML'] = self.res2/self.nObs
            
        return self._cache['sigma2_ML']       
    
    @property
    def std_res(self):  
        """
        standardised residuals
        
        Methods: p215, (9.7), Fotheringham, Brunsdon and Charlton (2002)
        """
        if 'std_res' not in self._cache:
            self._cache['std_res'] = self.res/(np.sqrt(self.sigma2 * (1.0 - self.influ)))
        return self._cache['std_res']  
    
    @property
    def std_err(self):  
        """
        standard errors of Betas
        
        Methods: p55, (2.15) and (2.21) , Fotheringham, Brunsdon and Charlton (2002)
        """
        if 'std_err' not in self._cache:
            self._cache['std_err'] = np.sqrt(self.CCT * self.sigma2)
        return self._cache['std_err']      
    
    @property
    def influ(self):
        """
        leading diagonal of S matrix
        """
        if 'influ' not in self._cache:
            self._cache['influ'] = np.reshape(np.diag(self.SMatrix),(-1,1))
        return self._cache['influ']  
    
    @property
    def CooksD(self):
        """
        Cook's D
        
        Methods: p216, (9.11), Fotheringham, Brunsdon and Charlton (2002)
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        if 'CooksD' not in self._cache:
            self._cache['CooksD'] = self.std_res**2 * self.influ /(self.tr_S * (1.0-self.influ))
        return self._cache['CooksD']     
    
    @property
    def t_stat(self):
        """
        t statistics of Beta
        """
        if 't_stat' not in self._cache:
            self._cache['t_stat'] = self.Betas *1.0/self.std_err
        return self._cache['t_stat']     
    
    @property
    def logll(self):
        """
        loglikelihood, put it here because it will be used to calculate other statistics
	
	Methods: p87 (4.2), Fotheringham, Brunsdon and Charlton (2002) 
	from Tomoki: log-likelihood = -0.5 *(double)N * (log(ss / (double)N * 2.0 * PI) + 1.0);
        """
        if 'logll' not in self._cache:
	    n = self.nObs
	    sigma2 = self.sigma2_ML   
            self._cache['logll'] = -0.5*n*(np.log(2*np.pi*sigma2)+1) 
        return self._cache['logll']     
    
    
    
class GWRGLM_Base(GWR_Base):
    """
    Basic class including common properties for GWGLM regression models
    
    Reference: 
    
    Nakaya, T., Fotheringham, S., Brunsdon, C. and Charlton, M. (2005): Geographically weighted Poisson regression for disease associative mapping, Statistics in Medicine 24, 2695-2717.
    
    Parameters
    ----------

    Attributes
    ----------
    """
    @property
    def y_bar(self):
        """
        weighted y mean
        """
        if 'y_bar' not in self._cache:
            arr_ybar = np.zeros(shape=(self.nObs,1)) 
            
            for i in range(self.nObs):
                w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                sum_yw = np.sum(self.y * w_i)
                if self.mType == 0 or self.mType == 2:
                    arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)    # weighted average  
                else:
                    arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * self.offset) 
            self._cache['y_bar'] = arr_ybar
        return self._cache['y_bar']    
    
    @property
    def sigma2_v1v2(self):  
        """
        residual variance
        
        Methods: p55 (2.16)-(2.18), Fotheringham, Brunsdon and Charlton (2002), use v1 and v2 #used in GWR4
        
        """
        if 'sigma2_v1v2' not in self._cache:            
            self._cache['sigma2_v1v2'] = self.res2/(self.nObs - 2.0*self.tr_S + self.tr_SWSTW) # method used in GWR4
            
        return self._cache['sigma2_v1v2']  
    
    #@property
    #def tr_S(self):  
        #"""
        #trace of S matrix
        
        #Methods: Nakaya et al. (2005): p2702, (30)
        #"""
        #if 'tr_S' not in self._cache:
            #self._cache['tr_S'] = np.trace(self.SMatrix)
        #return self._cache['tr_S']  
	
    @property
    def tr_SWSTW(self):  
	"""
	trace of STS matrix: S'WSW^-1
	"""
	if 'tr_SWSTW' not in self._cache:
	    w = np.reshape(self.w, (-1,1))
	    #stw = (self.SMatrix * 1.0/w).T
	    #sw = (self.SMatrix.T * w).T
	    #swstw = np.dot(sw, stw)
	    stw = (self.SMatrix * w).T
	    stws = np.dot(stw, self.SMatrix)
	    stwsw = stws.T *1.0/w
	    self._cache['tr_SWSTW'] = np.trace(stwsw) #np.trace(la.inv(swstw)) #np.trace(np.dot(self.SMatrix.T,self.SMatrix))
	return self._cache['tr_SWSTW']     
        
    @property
    def std_err(self):  
        """
        standard errors of Betas
        
        Methods: Nakaya et al. (2005): p2703, (32)
        """
        if 'std_err' not in self._cache:
	    if self.mType == 0: # Gaussian model
		self._cache['std_err'] = np.sqrt(self.CCT * self.sigma2)
	    else:
		self._cache['std_err'] = np.sqrt(self.CCT) # Poisson and logistic model
        return self._cache['std_err']       
    
    #@property, same basic GWR
    #def t_stat(self):
        #"""
        #t statistics of Beta
        #"""
        #if 't_stat' not in self._cache:
            #self._cache['t_stat'] = self.Betas *1.0/self.std_err
        #return self._cache['t_stat']  
    
    #@property
    def dev_res_GWGLM(self):
	"""
	get residual deviance of GLM model
	"""
	
	dev = np.zeros(shape=(self.nObs,1))
	#ones = np.ones(shape=(self.nObs,1)) 	
	#if self.mType == 2:
	    #v = np.log(self.y_pred*1.0/(1-self.y_pred))
	    #v1 = np.log(self.y_pred*1.0/1e-10)
	    
	for i in range(self.nObs):
	    w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))      
	    if self.mType == 1:
		for j in range(self.nObs):		
		    if self.y[j] <> 0:
			dev[i] += 2.0 * w_i[j] * self.y[j] *  np.log(self.y[j]/self.y_pred[j]) 
		    dev[i] -= 2.0 * (self.y[j] - self.y_pred[j]) * w_i[j]   
		    
	    if self.mType == 2:
		for j in range(self.nObs):
		    if self.y[j] == 0:
			dev[i] += -2.0 * np.log(1.0-self.y_pred[j]) * w_i[j]
		    else:
			dev[i] += -2.0 * np.log(self.y_pred[j]) * w_i[j]
       
	   
	return dev
    
    #@property
    def dev_mod_GWGLM(self):
	"""
	get model deviance of GWGLMMod model: for Poisson and Logistic
	
	model type, 0: Gaussian, 1: Poisson, 2: Logistic
	"""
	dev = np.zeros(shape=(self.nObs,1))
	ones = np.ones(shape=(self.nObs,1))
	for i in range(self.nObs):
	    w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))    
	    if self.mType == 1:	  
		for j in range(self.nObs):
		    if self.y[j] <> 0:   
			dev[i] += 2.0 * w_i[j] * self.y[j] * np.log(self.y[j]/(self.y_bar[i]*self.offset[j]))  
		    dev[i] -= 2.0 * (self.y[j] - self.y_bar[i] * self.offset[j]) * w_i[j]  
			
	    if self.mType == 2:	 
		for j in range(self.nObs):
		    if self.y[j] == 0:
			dev[i] += -2.0 * np.log(1.0-self.y_bar[i]) * w_i[j]
		    else:
			dev[i] += -2.0 * np.log(self.y_bar[i]) * w_i[j]		
	    
	return dev
    
    @property
    def localpDev(self):
	"""
	get local percent of deviance: for Poisson and Logistic
	
	localDev = 1.0 - dev_res / dev_mod
	"""
	if 'localpDev' not in self._cache:
	    ones = np.ones(shape=(self.nObs,1)) 
	    dev_res = self.dev_res_GWGLM() * 1.0
	    dev_mod = self.dev_mod_GWGLM()
	    localP = ones - dev_res/dev_mod
	    self._cache['localpDev'] = localP
		
	return self._cache['localpDev']
    
    @property
    def dev_res(self):
	"""
	get residual deviance of GLM model
	"""
	if 'dev_res' not in self._cache:
	    dev = 0.0
	    
	    if self.mType == 0:
		#res2 = np.sum(self.res**2)
		dev = self.nObs * (np.log(self.res2 * 2.0 * np.pi / self.nObs) + 1.0) # -2loglikelihood
		
	    if self.mType == 1:
		for i in range(self.nObs):
		    if (self.y[i] <> 0):
			dev += 2 * self.y[i] * np.log(self.y[i]/self.y_pred[i])
		    dev -= 2 * (self.y[i] - self.y_pred[i])
		dev = dev[0]
			
		    
	    if self.mType == 2:
		for i in range(self.nObs):
		    if self.y[i] == 0:
			dev += -2.0 * np.log(1.0 - self.y_pred[i])
		    else: 
			dev += -2.0 * np.log(self.y_pred[i])		
		#v = np.reshape(np.sum(self.x * self.Betas,axis=1),(-1,1))
		##v1 = 1.0
		
		#for i in range(self.nObs):
		    #y_pred  = 1.0/(1.0 + np.exp(-v[i]))
		    #if ((1.0 - y_pred) < 1e-10): 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1e-10))
		    #else: 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1.0 - y_pred))  	    
		dev = dev[0]	
		
	    self._cache['dev_res'] = dev
			
	return self._cache['dev_res']
    
    
class semiGWRGLM_Base(GWRGLM_Base):
    """
    Basic class including common properties for semiGWR regression models
    
    Reference: 
    
    Nakaya, T., Fotheringham, S., Brunsdon, C. and Charlton, M. (2005): Geographically weighted Poisson regression for disease associative mapping, Statistics in Medicine 24, 2695-2717.
    
    Parameters
    ----------

    Attributes
    ----------
    """
    #@property
    #def y_bar(self):
        #"""
        #weighted y mean
        #"""
        #if 'y_bar' not in self._cache:
            #arr_ybar = np.zeros(shape=(self.nObs,1)) 
            
            #for i in range(self.nObs):
                #w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                #sum_yw = np.sum(self.y * w_i)
                #if self.mType == 0 or self.mType == 2:
                    #arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)    # weighted average  
                #else:
                    #arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * self.offset) 
            #self._cache['y_bar'] = arr_ybar
        #return self._cache['y_bar']  
	
    @property
    def sigma2_v1v2(self):  
        """
        residual variance
        
        Methods: p55 (2.16)-(2.18), Fotheringham, Brunsdon and Charlton (2002), use v1 and v2 #used in GWR4
        
        """
        if 'sigma2_v1v2' not in self._cache:         
	    if self.mType == 0:
		self._cache['sigma2_v1v2'] = self.res2/(self.nObs - 2.0*self.tr_S + self.tr_STS) # method used in GWR4
            else:
		self._cache['sigma2_v1v2'] = self.res2/(self.nObs - 2.0*self.tr_S + self.tr_SWSTW)
		
        return self._cache['sigma2_v1v2']  
    
    @property
    def tr_S(self):  
        """
        trace of S matrix
        
        Methods: Nakaya et al. (2005): p2702, (30)
        """
        if 'tr_S' not in self._cache:
            self._cache['tr_S'] = self.m_glob.tr_S #np.trace(self.SMatrix)
        return self._cache['tr_S']  
    
    @property
    def tr_STS(self):  
	"""
	trace of STS matrix
	"""
	if 'tr_STS' not in self._cache:
	    self._cache['tr_STS'] = self.m_glob.tr_STS
	return self._cache['tr_STS']    
	
    @property
    def tr_SWSTW(self):  #???
	"""
	trace of STS matrix: S'WSW^-1
	"""
	if 'tr_SWSTW' not in self._cache:
	    #w = np.reshape(self.w, (-1,1))
	    #stw = (self.SMatrix * w).T
	    #stws = np.dot(stw, self.SMatrix)
	    #stwsw = stws.T *1.0/w
	    self._cache['tr_SWSTW'] = self.m_glob.tr_SWSTW #np.trace(stwsw) 
	return self._cache['tr_SWSTW']     
        
    @property
    def std_err_glob(self):  
        """
        standard errors of global Betas
        
        Methods: Nakaya et al. (2005): p2703, (47)
        """
        if 'std_err_glob' not in self._cache:
	    self._cache['std_err_glob'] = self.m_glob.std_err
        return self._cache['std_err_glob']   
    
    @property
    def std_err_loc(self):  
        """
        standard errors of local Betas
        
        Methods: Nakaya et al. (2005): p2703, (32)
        """
        if 'std_err_loc' not in self._cache:
	    if self.mType == 0: # Gaussian model
		if self.tr_S < self.tr_STS:
		    self.sigma2 = self.sigma2_v1
		else:
		    self.sigma2 = self.sigma2_v1v2
		self._cache['std_err_loc'] = np.sqrt(self.m_loc.CCT * self.sigma2)
	    else:
		self._cache['std_err_loc'] = np.sqrt(self.m_loc.CCT) # Poisson and logistic model
        return self._cache['std_err_loc']     
    
    @property 
    def t_stat_glob(self):
        """
        t statistics of global Beta
        """
        if 't_stat_glob' not in self._cache:
            self._cache['t_stat_glob'] = self.Betas_glob *1.0/self.std_err_glob
        return self._cache['t_stat_glob']  
    
    @property 
    def t_stat_loc(self):
        """
        t statistics of local Beta
        """
        if 't_stat_loc' not in self._cache:
            self._cache['t_stat_loc'] = self.Betas_loc *1.0/self.std_err_loc
        return self._cache['t_stat_loc']
    
    ##@property
    #def dev_res_GWGLM(self):
	#"""
	#get residual deviance of GLM model
	#"""
	
	#dev = np.zeros(shape=(self.nObs,1))
	##ones = np.ones(shape=(self.nObs,1)) 
	
	#if self.mType == 2:
	    #v = np.log(self.y_pred*1.0/(1-self.y_pred))
	    #v1 = np.log(self.y_pred*1.0/1e-10)
	    
	#for i in range(self.nObs):
	    #w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))      
	    #if self.mType == 1:
		##id0 = self.y==0
		##id1 = self.y<>0
		
		#if self.y[i] <> 0:
		    #dev[i] = 2.0 * np.sum(w_i * self.y *  np.log(self.y/self.y_pred)) 
		#dev[i] -= 2.0 * np.sum((self.y - self.y_pred) * w_i)   
		    
	    #if self.mType == 2:
		#dev_tmp = 0.0	    
		#for j in range(self.nObs):
		    #if ((1.0 - self.y_pred[j]) < 1e-10): 
			#dev_tmp += -2.0 * (self.y[j] * v1[i] + np.log(1e-10) ) * w_i[j]
		    #else: 
			#dev_tmp += -2.0 * (self.y[j] * v[i] + np.log(1.0 - self.y_pred[j]) ) * w_i[j]  
		#dev[i] = dev_tmp
       
	   
	#return dev
    
    ##@property
    #def dev_mod_GWGLM(self):
	#"""
	#get model deviance of GWGLMMod model: for Poisson and Logistic
	
	#model type, 0: Gaussian, 1: Poisson, 2: Logistic
	#"""
	#dev = np.zeros(shape=(self.nObs,1))
	#ones = np.ones(shape=(self.nObs,1))
	#for i in range(self.nObs):
	    #w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))    
	    #if self.mType == 1:	        
		##id0 = self.y==0
		##id1 = self.y<>0
		  
		#if self.y[i] <> 0:   
		    #dev[i] = 2.0 * np.sum( w_i * self.y * np.log(self.y/(self.y_bar[i]*self.offset)))  
		#dev[i] -= 2.0 * np.sum((self.y - self.y_bar[i] * self.offset) * w_i)  
		    
	    #if self.mType == 2:	    
		#v = np.log(self.y_bar[i]/(1.0 - self.y_bar[i]))
		#dev[i] = -2.0 * np.sum((self.y * v+ np.log(ones - self.y_bar[i])) * w_i)
	    
	#return dev
    
    @property
    def localpDev(self):
	"""
	get local percent of deviance: for Poisson and Logistic
	
	localDev = 1.0 - dev_res / dev_mod
	"""
	if 'localpDev' not in self._cache:
	    #ones = np.ones(shape=(self.nObs,1)) 
	    #dev_res = self.dev_res_GWGLM() * 1.0
	    #dev_mod = self.dev_mod_GWGLM()
	    #localP = self.m_loc.localpDev #ones - dev_res/dev_mod
	    self._cache['localpDev'] = self.m_loc.localpDev
		
	return self._cache['localpDev']
    
    @property
    def influ(self):
        """
        leading diagonal of S matrix
        """
        if 'influ' not in self._cache:
            self._cache['influ'] = self.m_glob.influ #np.reshape(np.diag(self.SMatrix),(-1,1))
        return self._cache['influ']  
    
    #@property
    #def localR2(self):
        #"""
        #local R square
        
        #Methods: p215, (9.8), Fotheringham, Brunsdon and Charlton (2002)
        #"""
        #if 'localR2' not in self._cache:
            #self._cache['localR2'] = (self.TSS() - self.RSS())/self.TSS() #(self.TSS - self.RSS)/self.TSS
        #return self._cache['localR2']   
    
    #@property
    #def dev_res(self):
	#"""
	#get residual deviance of GLM model
	#"""
	#if 'dev_res' not in self._cache:
	    #dev = 0.0
	    #if self.mType == 0:
		#res2 = np.sum(self.res**2)
		#dev = self.nObs * (np.log(res2 * 2.0 * np.pi / self.nObs) + 1.0) # -2loglikelihood
	    #if self.mType == 1:
		#for i in range(self.nObs):
		    #if (self.y[i] <> 0):
			#dev += 2 * self.y[i] * np.log(self.y[i]/self.y_pred[i])
		    #dev -= 2 * (self.y[i] - self.y_pred[i])
			
		    
	    #if self.mType == 2:
		#v = np.sum(self.x * self.Betas)
		##v1 = 1.0
		
		#for i in range(self.nObs):
		    #y_pred  = 1.0/(1.0 + np.exp(-v[i]))
		    #if ((1.0 - y_pred) < 1e-10): 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1e-10))
		    #else: 
			#dev += -2.0 * (self.y[i] * v[i] + np.log(1.0 - y_pred))   
	    
	    #dev = dev[0]		
	    #self._cache['dev_res'] = dev
			
	#return self._cache['dev_res']
    
    
    


    