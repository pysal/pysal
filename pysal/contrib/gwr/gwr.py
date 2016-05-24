import numpy as np
import numpy.linalg as la
from kernels import fix_gauss, fix_bisquare, fix_exp, adapt_gauss, adapt_bisquare, adapt_exp
import pysal.spreg.user_output as USER
from gwr_fits import gauss_iwls, poiss_iwls, logit_iwls
from families.family import Gaussian

class GWR(object):
    """
    Geographically weighted regression. Can currently estimate Gaussian,
    Poisson, and logistic models(built on a GLM framework). GWR object prepares
    model input. Fit method performs estimation and returns a GWRResults object.

    Parameters
    ----------
        y             : array
                        n*1, dependent variable
        x             : array
                        n*k, independent variable, exlcuding the constant
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'logistic'
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        this term is often the size of the population at risk or
                        the expected size of the outcome in spatial epidemiology
                        Default is None where Ni becomes 1.0 for all locations
        y_fix         : array
                        n*1, the fix intercept value of y
        sigma2_v1     : boolean
                        Sigma squared, True to use n as denominator
                        Default is False which uses n-k

    Attributes
    ----------
        y             : array
                        n*1, dependent variable
        x             : array
                        n*k, independent variable, including constant
        link          : string
                        Model type: 'Gaussian', 'Poisson', 'logistic'
        n             : integer
                        Number of observations
        k             : integer
                        Number of independent variables
        mean_y        : float
                        Mean of y
        std_y         : float
                        Standard deviation of y
        fit_params    : dict
                        Parameters passed into fit method to define estimation
                        routine
    """
    def __init__(self, coords, y, x, bw, family=Gaussian(), offset=None,
            y_fix=None, sigma2_v1=False, kernel='bisquare', fixed=False):
        """
        Initialize class
        """
        self.n = USER.check_arrays(y, x)
        USER.check_y(y, self.n)
        self.y = y
        self.x = USER.check_constant(x)
        self.family = family
        self.k = self.x.shape[1]
        self.sigma2_v1=sigma2_v1
        if offset is None:
            self.offset = np.ones(shape=(self.n,1))
        else:
            self.offset = offset * 1.0
        if y_fix is None:
	        self.y_fix = np.zeros(shape=(self.n,1))
        else:
            self.y_fix = y_fix
	    self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        self.fit_params = {}
        if fixed:
            if kernel == 'gaussian':
            	self.W = fix_gauss(coords, bw)
            elif kernel == 'bisquare':
                self.W = fix_bisquare(coords, bw)
            elif kernel == 'exponential':
                self.W = fix_exp(coords, bw)
            else:
                print 'Unsupported kernel function  ', kernel
        else:
            if kernel == 'gaussian':
            	self.W = adapt_gauss(coords, bw)
            elif kernel == 'bisquare':
                self.W = adapt_bisquare(coords, bw)
            elif kernel == 'exponential':
                self.W = adapt_exp(coords, bw)
            else:
                print 'Unsupported kernel function  ', kernel

    def fit(self, ini_betas=None, tol=1.0e-6, max_iter=200, solve='iwls'):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------

        ini_betas     : array
                        k*1, initial coefficient values, including constant.
                        Default is None, which calculates initial values during
                        estimation
        tol:            float
                        Tolerence for estimation convergence
        max_iter      : integer
                        Maximum number of iterations if convergence not
                        achieved
        solve         : string
                        Technique to solve MLE equations.
                        'iwls' = iteratively (re)weighted least squares (default)
        """
        self.fit_params['ini_betas'] = ini_betas
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']= solve
        if solve.lower() == 'iwls':
            betas = np.zeros((self.n, self.k))
            predy = np.zeros((self.n, 1))
            v = np.zeros((self.n, 1))
            w = np.zeros((self.n, 1))
            z = np.zeros((self.n, self.n))
            s = np.zeros((self.n, self.n))
            c = np.zeros((self.n, self.k))
            f = np.zeros((self.n, self.n))
            for i in range(self.n):
                wi = np.diag(self.W[i])
            	g_ey = self.family.link(self.y)
            	rslt = iwls(self.x, self.y, g_ey, self.family, self.offset,
            	        self.y_fix, self.fit_params['max_iter', wi=wi])
            	print rslt
                
                betas[i,:] = rslt[0]
                predy[i] = rslt[1][i]
                v[i] = rslt[2][i]
                w[i] = rslt[3][i]
                z[i] = rslt[4].flatten()
                ri = np.dot(self.x[i], rslt[5])
                s[i] = ri*np.reshape(rslt[4].flatten(), (1,-1))
                cf = rslt[5] - np.dot(rslt[5], f)
                c[i] =  np.diag(np.dot(cf, cf.T/w[i])).shape
            self.S = s*(1.0/z)
            self.CCT = c
        return GWRResults(self, betas, predy, v, w)

class GWRResults(GWR):
    """
    Basic class including common properties for all GWR regression models

    Parameters
    ----------
    model         : GWR object
                    Pointer to GWR object with estimation parameters.
    betas         : array
                    k*1, estimared coefficients
    predy         : array
                    n*1, predicted y values.
    v             : array
                    n*1, predicted y values before transformation via link.
    w             : array
                    n*1, final weight used for irwl

    Attributes
    ----------
    model         : GWR Object
                    Points to GWR object for which parameters have been
                    estimated.
    y             : array
                    n*1, dependent variable.
    x             : array
                    n*k, independent variable, including constant.
    family        : string
                    Model type: 'Gaussian', 'Poisson', 'Logistic'
    n             : integer
                    Number of observations
    k             : integer
                    Number of independent variables
    fit_params    : dict
                    Parameters passed into fit method to define estimation
                    routine.
    sig2          : float
                    sigma squared used for subsequent computations.
    betas         : array
                    n*k, Beta estimation
    w             : array
                    n*1, final weight used for x
    v             : array
                    n*1, untransformed predicted functions.
                    Applying the link functions yields predy.
    utu           : array
                    
    W             :
    
    S             :

    CCT           : 
    
    u             : array
                    n*1, residuals
    predy         : array
                    n*1, predicted value of y    
    tr_S          : float
                    trace of S (hat) matrix
    tr_STS        : float
                    trace of STS matrix
    y_bar         : array
                    n*1, weighted mean value of y
    TSS           : array
                    n*1, geographically weighted total sum of squares
    RSS           : array
                    n*1, geographically weighted residual sum of squares
    localR2       : array
                    n*1, local R square
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
    influ         : array
                    n*1, leading diagonal of S matrixi
    CooksD        : array
                    n*1, Cook's D
    t_stat        : array
                    n*k, local t-statistics
    logll         : float
                    log-likelihood
    dev_u         : float
                    deviance of residuals
    """
    def __init__(self, model, betas, predy, v=None, w=None):
        self.model = model
        self.n = model.n
        self.y = model.y
        self.x = model.x
        self.k = model.k
        self.family = model.family
        self.fit_params = model.fit_params
        self.betas = betas
        self.W = model.W
        if v is not None:
        	self.v = v
        if w is not None:
        	self.w = w
        self.S = model.S
        self.CCT = model.CCT
        self.predy = predy
        self.u = (self.y - self.predy).flatten()
        self.utu = np.dot(self.u, self.u.T)
        self._cache = {}

        if model.sigma2_v1:
        	self.sig2 = self.sigma2_v1
        else:
            self.sig2 = self.sigma2_v1v2

    @property
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        try:
            return self._cache['tr_S']
        except AttributeError:
            self._cache = {}
            self._cache['tr_S'] = np.trace(self.S)
        except KeyError:
            self._cache['tr_S'] = np.trace(self.S)
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
            self._cache['tr_STS'] = np.trace(np.dot(self.S.T,self.S))
        except KeyError:
	        self._cache['tr_STS'] = np.trace(np.dot(self.S.T,self.S))
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

            arr_ybar = np.zeros(shape=(self.n,1))
            for i in range(self.n):
                w_i= np.reshape(np.array(self.kernel.w[i]), (-1, 1))
                sum_yw = np.sum(self.y * w_i)
                arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)
            self._cache['y_bar'] = arr_ybar
        except KeyError:
            arr_ybar = np.zeros(shape=(self.n,1))
            for i in range(self.n):
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

        Methods: p215, (9.9)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        """
        try:
            return self._cache['TSS']
        except AttributeError:
            self._cache = {}
            arr_R = np.zeros(shape=(self.n,1))
            for i in range(self.n):
	            arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1)) *
	                    (self.y - self.y_bar[i])**2)
            self._cache['TSS'] = arr_R
        except KeyError:
    	    arr_R = np.zeros(shape=(self.n,1))
    	    for i in range(self.n):
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

        Methods: p215, (9.10)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        """
        try:
            return self._cache['RSS']
        except AttributeError:
            self._cache = {}
    	    arr_R = np.zeros(shape=(self.n,1))
            for i in range(self.n):
                arr_R[i] = np.sum(np.reshape(np.array(self.kernel.w[i]), (-1,1))
	                    * self.u**2)
            self._cache['RSS'] = arr_R
        except KeyError:
            arr_R = np.zeros(shape=(self.n,1))
            for i in range(self.n):
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

        Methods: p215, (9.8)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
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

        Methods: p214, (9.6),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        only use v1
        """
        try:
            return self._cache['sigma2_v1']
        except AttributeError:
            self._cache = {}
            self._cache['sigma2_v1'] = (self.utu/(self.n-self.tr_S))
        except KeyError:
            self._cache['sigma2_v1'] = (self.utu/(self.n-self.tr_S))
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

        Methods: p55 (2.16)-(2.18)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        use v1 and v2 #used in GWR4
        """
        try:
            return self._cache['sigma2_v1v2']
        except AttributeError:
            self._cache = {}
            self._cache['sigma2_v1v2'] = self.utu/(self.n - 2.0*self.tr_S +
	                self.tr_STS)
        except KeyError:
            self._cache['sigma2_v1v2'] = self.utu/(self.n - 2.0*self.tr_S +
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
            self._cache['sigma2_ML'] = self.utu/self.n
        except KeyError:
            self._cache['sigma2_ML'] = self.utu/self.n
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


    @property
    def std_res(self):
        """
        standardized residuals

        Methods:  p215, (9.7)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.


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

    @property
    def std_err(self):
        """
        standard errors of Betas

        Methods:  p215, (2.15) and (2.21)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.


        """
        try:
            return self._cache['std_err']
        except AttributeError:
            self._cache = {}
            self._cache['std_err'] = np.sqrt(self.CCT * self.sig2)
        except KeyError:
            self._cache['std_err'] = np.sqrt(self.CCT * self.sig2)
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

    @property
    def influ(self):
        """
        Influence: leading diagonal of S Matrix

        """
        try:
            return self._cache['influ']
        except AttributeError:
            self._cache = {}
            self._cache['influ'] = np.reshape(np.diag(self.S),(-1,1))
        except KeyError:
            self._cache['influ'] = np.reshape(np.diag(self.S),(-1,1))
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

    @property
    def cooksD(self):
        """
        Influence: leading diagonal of S Matrix

        Methods: p216, (9.11),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        try:
            return self._cache['cooksD']
        except AttributeError:
            self._cache = {}
            self._cache['cooksD'] = self.std_res**2 * self.influ / (self.tr_S * (1.0-self.influ))
        except KeyError:
            self._cache['cooksD'] =  self.std_res**2 * self.influ / (self.tr_S * (1.0-self.influ))
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

    @property
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

    @property
    def logll(self):
        """
        loglikelihood

	    Methods: p87 (4.2),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
	    from Tomoki: log-likelihood = -0.5 *(double)N * (log(ss / (double)N * 2.0 * PI) + 1.0);
        """
        try:
            return self._cache['log_ll']
        except AttributeError:
            self._cache = {}
            self._cache['logll'] = -0.5*self.n*(np.log(2*np.pi*self.sig2)+1)
        except KeyError:
            self._cache['logll'] = -0.5*self.n*(np.log(2*np.pi*self.sig2)+1)
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


    @property
    def dev_u(self):
        """
        deviance of residuals
        """
        try:
            return self._cache['dev_u']
        except AttributeError:
                self._cache = {}
                self._cache['dev_u'] = self.calc_dev_u()
        except KeyError:
            self._cache['dev_u'] = self.calc_dev_u()
        return self._cache['dev_u']

        @dev_u.setter
        def dev_u(self, val):
            try:
                self._cache['dev_u'] = val
            except AttributeError:
                self._cache = {}
                self._cache['dev_u'] = val
            except KeyError:
                self.cache['dev_u'] = val


    def calc_dev_u(self):
        dev = 0.0
        if self.family == 'Gaussian':
            dev = self.n * (np.log(self.utu * 2.0 * np.pi / self.n) + 1.0)
        if self.family == 'Poisson':
            id0 = self.y==0
            id1 = self.y<>0
            if np.sum(id1) == self.n:
                dev = 2.0 * np.sum(self.y * np.log(self.y/self.predy))
            else:
                dev = 2.0 * (np.sum(self.y[id1] *
                        np.log(self.y[id1]/self.predy[id1])) -
                            np.sum(self.y[id0]-self.predy[id0]))
        if self.family == 'logistic':
            for i in range(self.n):
                if self.y[i] == 0:
                    dev += -2.0 * np.log(1.0 - self.predy[i])
                else:
                    dev += -2.0 * np.log(self.predy[i])
        return dev

