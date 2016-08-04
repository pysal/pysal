import numpy as np
import numpy.linalg as la
from kernels import fix_gauss, fix_bisquare, fix_exp, adapt_gauss, adapt_bisquare, adapt_exp
import pysal.spreg.user_output as USER
import sys
sys.path.append('/Users/toshan/dev/pysal/pysal/contrib/glm/')
from family import Gaussian, Binomial, Poisson
from glm import GLM, GLMResults
from iwls import iwls
from utils import cache_readonly

class GWR(GLM):
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
    def __init__(self, coords, y, X, bw, family=Gaussian(), offset=None,
            y_fix=None, sigma2_v1=False, kernel='bisquare', fixed=False,
            constant=True):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, offset, y_fix, constant)
        self.sigma2_v1=sigma2_v1
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

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls'):
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
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']= solve
        if solve.lower() == 'iwls':
            params = np.zeros((self.n, self.k))
            predy = np.zeros((self.n, 1))
            v = np.zeros((self.n, 1))
            w = np.zeros((self.n, 1))
            z = np.zeros((self.n, self.n))
            s = np.zeros((self.n, self.n))
            c = np.zeros((self.n, self.k))
            f = np.zeros((self.n, self.n))
            p = np.zeros((self.n, 1))
            dev_resa = np.zeros((self.n, 1))
            dev = np.zeros((self.n, 1))
            for i in range(self.n):
                wi = np.diag(self.W[i])
            	rslt = iwls(self.y, self.X, self.family, self.offset,
            	        self.y_fix, ini_params, tol, max_iter, wi=wi)
                
                params[i,:] = rslt[0].T
                
                predy[i] = rslt[1][i]
                v[i] = rslt[2][i]
                w[i] = rslt[3][i]
                z[i] = rslt[4].flatten()
                ri = np.dot(self.X[i], rslt[5])
                s[i] = ri*np.reshape(rslt[4].flatten(), (1,-1))
                cf = rslt[5] - np.dot(rslt[5], f)
                c[i] = np.diag(np.dot(cf, cf.T/1))
                #sign = np.sign(self.y-rslt[1])
                #dev_res = sign * 2.0 * wi * (self.y * np.log(self.y/rslt[1]) -
                #        (self.y - rslt[1]))
                #dev = 2.0 * np.sum(wi*np.log(self.y/rslt[1]))
                #dev_res[i] = ((self.family.resid_dev(self.y,rslt[1]))**2).flatten()
                #dev[i] = self.family.deviance(self.y, rslt[1])
                #p[i] = 1.0 - (np.sum(np.abs(dev_res)) / dev)
            for i in range(self.n):
                wi = np.diag(self.W[i])
                dev_resa[i] = np.sum(((self.family.resid_dev(self.y,predy))**2)*wi)

            #dev_res = np.zeros((self.n, self.n))
            #global_dev_res = ((self.family.resid_dev(self.y,predy))**2)
            #global_dev = self.family.deviance(self.y, predy)
            #dev_res = np.repeat(global_dev_res.flatten(),self.n)
            #dev_res = dev_res.reshape((self.n, self.n))
            #dev_res = np.sum(dev_res * self.W.T, axis=0)
            #dev = (self.W * global_dev) - (self.W*(2.0 * (self.y - self.y_bar)))
            #dev = np.sum(dev, axis=0)
            #dev  = dev/self.n
            self.f = f
            self.cf = cf
            self.S = s*(1.0/z)
            self.CCT = c
            #self.pDev = 1 - (np.sum(dev_res)/ dev)
            #print self.pDev
            #if isinstance(self.family, Poisson):
                #print dev[0]
        return GWRResults(self, params, predy, v, w)

class GWRResults(GLMResults):
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
    def __init__(self, model, params, predy, v=None, w=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.W = model.W
        if v is not None:
        	self.v = v
        if w is not None:
        	self.w = w
        self.S = model.S
        self.CCT = model.CCT
        #self.pDev  = model.pDev
        self.u = (self.resid_response).flatten()
        #self.u = self.u.reshape((-1,1))
        self.utu = np.dot(self.u, self.u.T)
        self.u = self.u.reshape((-1,1))
        self._cache = {}
        if model.sigma2_v1:
        	self.sig2 = self.sigma2_v1
        else:
            self.sig2 = self.sigma2_v1v2

    @cache_readonly
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        return np.trace(self.S)

    @cache_readonly
    def tr_STS(self):
        """
        trace of STS matrix
        """
        return np.trace(np.dot(self.S.T,self.S))

    @cache_readonly
    def y_bar(self):
        """
        weighted mean of y
        """
        arr_ybar = np.zeros(shape=(self.n,1))
        for i in range(self.n):
            w_i= np.reshape(np.array(self.W[i]), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1,1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i)
        return arr_ybar

    @cache_readonly
    def TSS(self):
        """
        geographically weighted total sum of squares

        Methods: p215, (9.9)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        """
        TSS = np.zeros(shape=(self.n,1))
        for i in range(self.n):
	        TSS[i] = np.sum(np.reshape(np.array(self.W[i]), (-1,1)) *
	                (self.y.reshape((-1,1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):
        """
        geographically weighted residual sum of squares

        Methods: p215, (9.10)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        """
    	RSS = np.zeros(shape=(self.n,1))
        for i in range(self.n):
            RSS[i] = np.sum(np.reshape(np.array(self.W[i]), (-1,1))
	                * self.u**2)
        return RSS

    @cache_readonly
    def localR2(self):
        """
        local R square

        Methods: p215, (9.8)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        """
        return (self.TSS - self.RSS)/self.TSS

    @cache_readonly
    def sigma2_v1(self):
        """
        residual variance

        Methods: p214, (9.6),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        only use v1
        """
        return (self.utu/(self.n-self.tr_S))
    
    @cache_readonly
    def sigma2_v1v2(self):
        """
        residual variance

        Methods: p55 (2.16)-(2.18)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.

        use v1 and v2 #used in GWR4
        """
        return self.utu/(self.n - 2.0*self.tr_S +
	                self.tr_STS)

    @cache_readonly
    def sigma2_ML(self):
        """
        residual variance

        Methods: maximum likelihood
        """
        return self.utu/self.n

    @cache_readonly
    def std_res(self):
        """
        standardized residuals

        Methods:  p215, (9.7)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        """
        return self.u.reshape((-1,1))/(np.sqrt(self.sig2 * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        """
        standard errors of Betas

        Methods:  p215, (2.15) and (2.21)
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        """
        if isinstance(self.family, (Poisson, Binomial)):
            return np.sqrt(self.CCT)
        else:
            return np.sqrt(self.CCT*self.sig2)

    @cache_readonly
    def influ(self):
        """
        Influence: leading diagonal of S Matrix

        """
        return np.reshape(np.diag(self.S),(-1,1))

    @cache_readonly
    def cooksD(self):
        """
        Influence: leading diagonal of S Matrix

        Methods: p216, (9.11),
        Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
        Geographically weighted regression: the analysis of spatially varying relationships.
        Note: in (9.11), p should be tr(S), that is, the effective number of parameters
        """
        return self.std_res**2 * self.influ / (self.tr_S * (1.0-self.influ))

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

    @cache_readonly
    def pDev(self):
        global_dev_res = ((self.family.resid_dev(self.y,self.mu))**2)
        dev_res = np.repeat(global_dev_res.flatten(),self.n)
        dev_res = dev_res.reshape((self.n, self.n))
        dev_res = np.sum(dev_res * self.W.T, axis=0)
        if isinstance(self.family, Gaussian):
        	return np.nan
        elif isinstance(self.family, Poisson):
            dev = np.sum(2.0*self.W*(self.y*np.log(self.y/(self.y_bar))-(self.y-self.y_bar)),axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return  1.0 - (dev_res.reshape((-1,1))/ dev.reshape((-1,1)))
