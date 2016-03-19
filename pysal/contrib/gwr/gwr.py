import numpy as np
import numpy.linalg as la
from utils import RegressionPropsY, RegressionPropsVM


    
class GWR_Base(RegressionPropsY, RegressionPropsVM):
    """
    Basic class including common properties for all GWR regression models
    
    Parameters
    ----------

    Attributes
    ----------
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
                n*1, leading diagonal of S matrixi
    logll     :
    """
      

    @property
    def tr_S(self):  
        """
        trace of S matrix
        """
        try:
            return self._cache['tr_S']
        except AttributeError:
            self._cache = {}
            self._cache['tr_S'] = np.trace(self.SMatrix)
        except KeyError:
            self._cache['tr_S'] = np.trace(self.SMatrix)
        return self._cache['tr_S']   
    
    @tr_S.setter
    def tr_S(self, val):
        try: 
            self._cache['tr_S'] = val
        except AttributeError:
            self._cache = {}
            self._cache['tr_S'] = val
        except KeyError:
            self._cache['tr_S'] = val

    @property
    def tr_STS(self):  
        """
        trace of STS matrix
        """
        try:
            return self._cache['tr_STS']
        except AttributeError:
            self._cache = {}          
	        self._cache['tr_STS'] = np.trace(np.dot(self.SMatrix.T,self.SMatrix))
        except KeyError:
	        self._cache['tr_STS'] = np.trace(np.dot(self.SMatrix.T,self.SMatrix))
        return self._cache['tr_STS']   
    
    @tr_STS.setter
    def tr_STS(self, val):
        try: 
            self._cache['tr_STS'] = val
        except AttributeError:
            self._cache = {}
            self._cache['tr_STS'] = val
        except KeyError:
            self._cache['tr_STS'] = val


    @property
    def y_bar(self):  
        """
        weighted mean of y
        """
        try:
            return self._cache['y_bar']
        except AttributeError:
            self._cache = {}
            
            arr_ybar = np.zeros(shape=(self.nObs,1))            
            for i in range(self.nObs):
                w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                sum_yw = np.sum(self.y * w_i)
                arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)              
            self._cache['y_bar'] = arr_ybar
        except KeyError:
            arr_ybar = np.zeros(shape=(self.nObs,1))            
            for i in range(self.nObs):
                w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                sum_yw = np.sum(self.y * w_i)
                arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)
            self._cache['y_bar'] = arr_ybar
        return self._cache['y_bar']   
    
    @y_bar.setter
    def y_bar(self, val):
        try: 
            self._cache['y_bar'] = val
        except AttributeError:
            self._cache = {}
            self._cache['y_bar'] = val
        except KeyError:
            self._cache['y_bar'] = val
        
    @property
    def TSS(self):  
        """
        geographically weighted total sum of squares
        
        Methods: p215, (9.9), Fotheringham, Brunsdon and Charlton (2002)
        """
        try:
            return self._cache['TSS']
        except AttributeError:
            self._cache = {}          
    	    arr_R = np.zeros(shape=(self.nObs,1))
	        for i in range(self.nObs):
	            arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1)) *
	                    (self.y - self.y_bar[i])**2)               
	        self._cache['TSS'] = arr_R
        except KeyError:
    	    arr_R = np.zeros(shape=(self.nObs,1))
	        for i in range(self.nObs):
	            arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1)) *
	                    (self.y - self.y_bar[i])**2)               
	        self._cache['TSS'] = arr_R
        return self._cache['TSS']   
    
    @TSS.setter
    def TSS(self, val):
        try: 
            self._cache['TSS'] = val
        except AttributeError:
            self._cache = {}
            self._cache['TSS'] = val
        except KeyError:
            self._cache['TSS'] = val
    

    @property
    def RSS(self):  
        """
        geographically weighted residual sum of squares
        
        Methods: p215, (9.10), Fotheringham, Brunsdon and Charlton (2002)
        """
        try:
            return self._cache['RSS']
        except AttributeError:
            self._cache = {}          
    	    arr_R = np.zeros(shape=(self.nObs,1))
	        for i in range(self.nObs):
	            arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1))
	                    * self.u**2)               
	        self._cache['RSS'] = arr_R
        except KeyError:
    	    arr_R = np.zeros(shape=(self.nObs,1))
	        for i in range(self.nObs):
	            arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1))
	                    * self.u**2)               
	        self._cache['RSS'] = arr_R
        return self._cache['RSS']   
    
    @RSS.setter
    def RSS(self, val):
        try: 
            self._cache['RSS'] = val
        except AttributeError:
            self._cache = {}
            self._cache['RSS'] = val
        except KeyError:
            self._cache['RSS'] = val



    @property
    def localR2(self):  
        """
        local R square
        
        Methods: p215, (9.8), Fotheringham, Brunsdon and Charlton (2002)
        """
        try:
            return self._cache['localR2']
        except AttributeError:
            self._cache = {}          
	        self._cache['localR2'] = (self.TSS() - self.RSS())/self.TSS() 
        except KeyError:
	        self._cache['localR2'] = (self.TSS() - self.RSS())/self.TSS() 
        return self._cache['localR2']   
    
    @localR2.setter
    def localR2(self, val):
        try: 
            self._cache['localR2'] = val
        except AttributeError:
            self._cache = {}
            self._cache['localR2'] = val
        except KeyError:
            self._cache['localR2'] = val


    @property
    def sigma2_v1(self):  
        """
        residual variance
        
        Methods: p214, (9.6), Fotheringham, Brunsdon and Charlton (2002),
        only use v1
        """
        try:
            return self._cache['sigma2_v1']
        except AttributeError:
            self._cache = {}          
	        self._cache['sigma2_v1'] = (self.utu/(self.nObs-self.tr_S)) 
        except KeyError:
	        self._cache['sigma2_v1'] = (self.utu/(self.nObs-self.tr_S)) 
        return self._cache['sigma2_v1']   
    
    @sigma2_v1.setter
    def sigma2_v1(self, val):
        try: 
            self._cache['sigma2_v1'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sigma2_v1'] = val
        except KeyError:
            self._cache['sogma2_v1'] = val
    

    @property
    def sigma2_v1v2(self):  
        """
        residual variance
        
        Methods: p55 (2.16)-(2.18), Fotheringham, Brunsdon and Charlton (2002), 
        use v1 and v2 #used in GWR4
        """
        try:
            return self._cache['sigma2_v1v2']
        except AttributeError:
            self._cache = {}          
	        self._cache['sigma2_v1v2'] = self.utu/(self.nObs - 2.0*self.tr_S + 
	                self.tr_STS)  
        except KeyError:
	        self._cache['sigma2_v1v2'] = self.utu/(self.nObs - 2.0*self.tr_S + 
	                self.tr_STS)  
        return self._cache['sigma2_v1v2']   
    
    @sigma2_v1v2.setter
    def sigma2_v1v2(self, val):
        try: 
            self._cache['sigma2_v1v2'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sigma2_v1v2'] = val
        except KeyError:
            self._cache['sogma2_v1v2'] = val
    

    @property
    def sigma2_ML(self):  
        """
        residual variance
        
        Methods: maximum likelihood 
        """
        try:
            return self._cache['sigma2_ML']
        except AttributeError:
            self._cache = {}          
	        self._cache['sigma2_ML'] = self.utu/self.nObs 
        except KeyError:
	        self._cache['sigma2_ML'] = self.utu/self.nObs
        return self._cache['sigma2_ML']   
    
    @sigma2_ML.setter
    def sigma2_ML(self, val):
        try: 
            self._cache['sigma2_ML'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sigma2_ML'] = val
        except KeyError:
            self._cache['sigma2_ML'] = val
    
    
    
    def std_res(self):  
        """
        standardized residuals

        Methods:  p215, (9.7), Fotheringham Brundson and Charlton (2002)
        
        """
        try:
            return self._cache['std_res']
        except AttributeError:
            self._cache = {}          
	        self._cache['std_res'] = self.u/(np.sqrt(self.sigma2 * (1.0 - self.influ)))
        except KeyError:
	        self._cache['std_res'] = self.u/(np.sqrt(self.sigma2 * (1.0 - self.influ)))
        return self._cache['std_res']   
    
    @std_res.setter
    def std_res(self, val):
        try: 
            self._cache['std_res'] = val
        except AttributeError:
            self._cache = {}
            self._cache['std_res'] = val
        except KeyError:
            self._cache['std_res'] = val
    
   
    def std_err(self):  
        """
        standard errors of Betas

        Methods:  p215, (2.15) and (2.21), Fotheringham Brundson and Charlton (2002)
        
        """
        try:
            return self._cache['std_err']
        except AttributeError:
            self._cache = {}          
	        self._cache['std_err'] = np.sqrt(self.CCT * self.sigma2)
        except KeyError:
	        self._cache['std_err'] = np.sqrt(self.CCT * self.sigma2)
        return self._cache['std_err']   
    
    @std_err.setter
    def std_err(self, val):
        try: 
            self._cache['std_err'] = val
        except AttributeError:
            self._cache = {}
            self._cache['std_err'] = val
        except KeyError:
            self._cache['std_err'] = val

    def influ(self):  
        """
        Influence: leading diagonal of S Matrix

        """
        try:
            return self._cache['influ']
        except AttributeError:
            self._cache = {}          
	        self._cache['influ'] = np.reshape(np.diag(self.SMatrix),(-1,1))
        except KeyError:
	        self._cache['influ'] = np.reshape(np.diag(self.SMatrix),(-1,1))
        return self._cache['influ']   
    
    @influ.setter
    def influ(self, val):
        try: 
            self._cache['influ'] = val
        except AttributeError:
            self._cache = {}
            self._cache['influ'] = val
        except KeyError:
            self._cache['influ'] = val
    
    def cooksD(self):  
        """
        Influence: leading diagonal of S Matrix

        Methods: p216, (9.11), Fotheringham, Brunsdon and Charlton (2002)
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        try:
            return self._cache['cooksD']
        except AttributeError:
            self._cache = {}          
	        self._cache['cooksD'] = self.std_res**2 * self.influ /
	        (self.tr_S * (1.0-self.influ)) 
        except KeyError:
	        self._cache['cooksD'] =  self.std_res**2 * self.influ /
	        (self.tr_S * (1.0-self.influ))
        return self._cache['cooksD']   
    
    @cooksD.setter
    def cooksD(self, val):
        try: 
            self._cache['cooksD'] = val
        except AttributeError:
            self._cache = {}
            self._cache['cooksD'] = val
        except KeyError:
            self._cache['cooksD'] = val
    
        
    def t_stat(self):  
        """
        t statistics of Betas

        """
        try:
            return self._cache['t_stat']
        except AttributeError:
            self._cache = {}          
	        self._cache['t_stat'] = self.Betas *1.0/self.std_err
        except KeyError:
	        self._cache['t_stat'] = self.Betas *1.0/self.std_err 
        return self._cache['t_stat']   
    
    @t_stat.setter
    def t_stat(self, val):
        try: 
            self._cache['t_stat'] = val
        except AttributeError:
            self._cache = {}
            self._cache['t_stat'] = val
        except KeyError:
            self._cache['t_stat'] = val
   

    def logll(self):  
        """

        loglikelihood, put it here because it will be used to calculate other statistics
	
	    Methods: p87 (4.2), Fotheringham, Brunsdon and Charlton (2002) 
	    from Tomoki: log-likelihood = -0.5 *(double)N * (log(ss / (double)N * 2.0 * PI) + 1.0);
        """
        try:
            return self._cache['log_ll']
        except AttributeError:
            self._cache = {}          
	        self._cache['logll'] = -0.5*n*(np.log(2*np.pi*sigma2)+1)
        except KeyError:
	        self._cache['logll'] = -0.5*n*(np.log(2*np.pi*sigma2)+1) 
        return self._cache['logll']   
    
    @logll.setter
    def logll(self, val):
        try: 
            self._cache['logll'] = val
        except AttributeError:
            self._cache = {}
            self._cache['logll'] = val
        except KeyError:
            self._cache['logll'] = val
    
    
    
