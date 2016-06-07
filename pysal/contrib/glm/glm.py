#TODO
# Add model diagnostics as cached properties:
# Add family class functionality so that diagnostics are methods of family class
# intead of using different cases for family for each diagnostic.

import numpy as np
import numpy.linalg as la
from family import Gaussian, Binomial, Poisson
from pysal.spreg.utils import RegressionPropsY
from iwls import iwls
import pysal.spreg.user_output as USER
from utils import np_matrix_rank, cache_readonly

__all__ = ['GLM']

class GLM(RegressionPropsY):
    """
    Generalised linear models. Can currently estimate Guassian, Poisson and
    Logisitc regression coefficients. GLM object prepares model input and fit
    method performs estimation which then returns a GLMResults object.

    Parameters
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, exlcuding the constant.
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'Binomial'
        offset        : array
                        n*1, the offset variable at the ith location. For Poisson model
                        this term is often the size of the population at risk or
                        the expected size of the outcome in spatial epidemiology.
                        Default is None where Ni becomes 1.0 for all locations.
        y_fix         : array
                        n*1, the fix intercept value of y
        sigma2_v1     : boolean
                        Sigma squared, True to use n as denominator.
                        Default is False which uses n-k.

    Attributes
    ----------
        y             : array
                        n*1, dependent variable.
        x             : array
                        n*k, independent variable, including constant.
        family        : string
                        Model type: 'Gaussian', 'Poisson', 'logistic'
        n             : integer
                        Number of observations
        k             : integer
                        Number of independent variables
        mean_y        : float
                        Mean of y
        std_y         : float
                        Standard deviation of y
        fit_params     : dict
                        Parameters passed into fit method to define estimation
                        routine.
    """
    def __init__(self, y, X, family=Gaussian(), offset=None, y_fix = None,
            sigma2_v1=False, constant=True):
        """
        Initialize class
        """
        self.n = USER.check_arrays(y, X)
        USER.check_y(y, self.n)
        self.y = y
        if constant:
            self.X = USER.check_constant(X)
        else:
            self.X = X
        self.family = family
        self.k = X.shape[1]
        self.df_model = np_matrix_rank(self.X) - 1
        self.df_resid = self.n - self.df_model - 1
        self.sigma2_v1=sigma2_v1
        if offset is None:
            self.offset = np.ones(shape=(self.n,1))
        else:
            self.offset = offset * 1.0
        if y_fix is None:
	        self.y_fix = np.zeros(shape=(self.n,1))
        else:
	        self.y_fix = y_fix
        self.fit_params = {}


    def fit(self, ini_betas=None, tol=1.0e-6, max_iter=200, solve='iwls'):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------

        ini_betas     : array
                        k*1, initial coefficient values, including constant.
                        Default is None, which calculates initial values during
                        estimation.
        tol:            float
                        Tolerence for estimation convergence.
        max_iter       : integer
                        Maximum number of iterations if convergence not
                        achieved.
        solve         :string
                       Technique to solve MLE equations.
                       'iwls' = iteratively (re)weighted least squares (default)
        """
        self.fit_params['ini_betas'] = ini_betas
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']=solve
        if solve.lower() == 'iwls':
            params, predy, w, n_iter = iwls(self.y, self.X, self.family, self.offset, 
                    self.y_fix, ini_betas, tol, max_iter)
            self.fit_params['n_iter'] = n_iter
        return GLMResults(self, params, predy, w)


class GLMResults(GLM):
    """
    Results of estimated GLM and diagnostics.

    Parameters
    ----------
        model         : GLM object
                        Pointer to GLM object with estimation parameters.
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
        model         : GLM Object
                        Points to GLM object for which parameters have been
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
        xtxi          : array
                        n*k, inverse of xx' for computing covariance
        u             : array
                        n*1, residuals
        predy         : array
                        n*1, predicted value of y
        utu           : float
                        Sum of squared residuals
        sig2n         : float
                        sigma sqaured using n for denominator
        sig2n_k       : float
                        sigma sqaured using n-k for denominator
        vm            : array
                        Variance covariance matrix (kxk) of betas
        std_err       : array
                        k*1, standard errors of betas
        dev_u         : float
                        Deviance of residuals
        logll         : float
                        log-likelihood
        aic           : float 
                        AIC
        aicc          : float 
                        AICc
        bic           : float 
                        BIC
        cv            : float
                        CV
        R2            : float
                        R square
        y_bar         : array
                        n*1, mean value of y
    """

    def __init__(self, model, params, predy, w):
        self.model = model
        self.n = model.n
        self.y = model.y
        self.X = model.X
        self.k = model.k
        self.offset = model.offset
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.family = model.family
        self.fit_params = model.fit_params
        self.params = params
        self.w = w
        self.predy = predy
        self.u = self.y - self.predy
        self.xtxi = la.inv(np.dot(self.X.T,self.X))
        self._cache = {}

        #if model.sigma2_v1:
	        #self.sig2 = self.sig2n
        #else:
            #self.sig2 = self.sig2n_k

    @cache_readonly
    def resid_response(self):
        return self.y-self.mu

    @cache_readonly
    def resid_pearson(self):
        return  ((self.y-self.mu) /
                np.sqrt(self.family.variance(self.mu)))

    @cache_readonly
    def resid_working(self):
        return (self.resid_response / self.family.link.deriv(self.mu))

    @cache_readonly
    def resid_anscombe(self):
        return self.family.resid_anscombe(self.y, self.fittedvalues)

    @cache_readonly
    def resid_deviance(self):
        return self.family.resid_dev(self.y, self.fittedvalues)

    @cache_readonly
    def pearson_chi2(self):
        chisq = (self.y - self.mu)**2 / self.family.variance(self.mu)
        chisqsum = np.sum(chisq)
        return chisqsum


    @cache_readonly
    def fittedvalues(self):
        return self.mu


    @cache_readonly
    def mu(self):
        if np.allclose(self.offset, np.ones((self.n, 1))):
            linpred = np.dot(self.X, self.params)
        else:
            linepred = np.dot(self.X, self.params) + self.offset
        return self.family.fitted(linpred)

    @cache_readonly
    def null(self):
        y = self.y
        model = self.model
        X = np.ones((len(y), 1))
        null_mod =  GLM(y, X, family=self.family, offset=self.offset, constant=False)
        return null_mod.fit().fittedvalues
   
    @cache_readonly
    def scale(self):
        if isinstance(self.family, (Binomial, Poisson)):
            return 1.
        else:
            return (((np.power(self.resid_response, 2) /
                         self.family.variance(self.mu))).sum() /
                        (self.df_resid))

    @cache_readonly
    def deviance(self):
        return self.family.deviance(self.y, self.mu)

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)
   
    @cache_readonly
    def llnull(self):
        return self.family.loglike(self.y, self.null)

    @cache_readonly
    def llf(self):
        _modelfamily = self.family
        val = _modelfamily.loglike(self.y, self.mu)
        return val
    
    @cache_readonly
    def aic(self):
        return -2 * self.llf + 2*(self.df_model+1)

    @cache_readonly
    def bic(self):
        return (self.deviance -
                (self.model.n - self.df_model - 1) *
                np.log(self.model.n))
'''
    @property
    def utu(self):
        try:
            return self._cache['utu']
        except AttributeError:
            self._cache = {}
            self._cache['utu'] = np.sum(self.u ** 2)
        except KeyError:
            self._cache['utu'] = np.sum(self.u ** 2)
        return self._cache['utu']

    @utu.setter
    def utu(self, val):
        try:
            self._cache['utu'] = val
        except AttributeError:
            self._cache = {}
            self._cache['utu'] = val
        except KeyError:
            self._cache['utu'] = val

    @property
   def sig2n(self):
        try:
            return self._cache['sig2n']
        except AttributeError:
            self._cache = {}
            self._cache['sig2n'] = np.sum(self.w*self.u**2) / self.n
        except KeyError:
            self._cache['sig2n'] = np.sum(self.w*self.u**2) / self.n
        return self._cache['sig2n']

    @sig2n.setter
    def sig2n(self, val):
        try:
            self._cache['sig2n'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sig2n'] = val
        except KeyError:
            self._cache['sig2n'] = val

    @property
    def sig2n_k(self):
        try:
            return self._cache['sig2n_k']
        except AttributeError:
            self._cache = {}
            self._cache['sig2n_k'] = np.sum(self.w*self.u**2) / (self.n - self.k)
        except KeyError:
            self._cache['sig2n_k'] = np.sum(self.w*self.u**2) / (self.n - self.k)
        return self._cache['sig2n_k']

    @sig2n_k.setter
    def sig2n_k(self, val):
        try:
            self._cache['sig2n_k'] = val
        except AttributeError:
            self._cache = {}
            self._cache['sig2n_k'] = val
        except KeyError:
            self._cache['sig2n_k'] = val

    @property
    def vm(self):
        try:
            return self._cache['vm']
        except AttributeError:
            self._cache = {}
            if self.mType == 0:
        		self._cache['vm'] = np.dot(self.sig2, self.xtxi)
            else:
        	    xtw = (self.X * self.w).T
        	    xtwx = np.dot(xtw, self.X)
        	    self._cache['vm'] = la.inv(xtwx)
        except KeyError:
            if self.family == 'Gaussian':
        		self._cache['vm'] = np.dot(self.sig2, self.xtxi)
            else:
        	    xtw = (self.X * self.w).T
        	    xtwx = np.dot(xtw, self.X)
        	    self._cache['vm'] = la.inv(xtwx)
        return self._cache['vm']

    @vm.setter
    def vm(self, val):
        try:
            self._cache['vm'] = val
        except AttributeError:
            self._cache = {}
            self._cache['vm'] = val
        except KeyError:
            self._cache['vm'] = val

    @property
    def std_err(self):

        try:
            return self._cache['std_err']
        except AttributeError:
            self._cache = {}
            self._cache['std_err'] = np.sqrt(self.vm).diagonal()
        except KeyError:
            self._cache['std_err'] = np.sqrt(self.vm).diagonal()
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
    def dev_u(self):
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

   

    @property
    def y_bar(self):
        """
        mean of y
        """
        try:
            return self._cache['y_bar']
        except AttributeError:
            self._cache = {}
            self._cache['y_bar'] = np.sum(self.y)/n
        except KeyError:
            self._cache['y_bar'] = np.sum(self.y)/n
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
    def r2(self):
        try:
            return self._cache['r2']
        except AttributeError:
            self._cache = {}
            self._cache['r2'] = 1- self.utu/(np.sum((self.y-self.y_bar)**2))
        except KeyError:
            self._cache['r2'] = 1- self.utu/(np.sum((self.y-self.y_bar)**2))
        return self._cache['r2']

    @std_err.setter
    def r2(self, val):
        try:
            self._cache['r2'] = val
        except AttributeError:
            self._cache = {}
            self._cache['r2'] = val
        except KeyError:
            self._cache['r2'] = val
'''
