import numpy as np
import numpy.linalg as la
from kernels import fix_gauss, fix_bisquare, fix_exp, adapt_gauss, adapt_bisquare, adapt_exp
import pysal.spreg.user_output as USER
from glm_fits import gauss_iwls, poiss_iwls, logit_iwls

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
        y_fix         :

        sigma2_v1     : boolean
                        Sigma squared, True to use n as denominator
                        Default is False which uses n-k
        sMatrix       : array
                        n*n, hat matrix. Default is None

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
        fit_params     : dict
                        Parameters passed into fit method to define estimation
                        routine
    """
    def __init__(self, coords, y, x, bw, family='Gaussian', offset=None,
            y_fix=None, sigma2_v1=False, kernel='bisquare', fixed=False,
            sMatrix=None):
        """
        Initialize class
        """
        print(y.shape, x.shape)
        print(y)
        print(x)
        self.n = USER.check_arrays(y, x)
        USER.check_y(y, self.n)
        self.y = y
        self.x = USER.check_constant(x)
        self.sMatrix = sMatrix
        self.family = family
        self.k = x.shape[1]
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
        print bw
        if fixed:
            if kernel == 'gaussian':
            	self.wi = fix_gauss(coords, bw)
            elif kernel == 'bisquare':
                self.wi = fix_bisquare(coords, bw)
            elif kernel == 'exp':
                self.wi = fix_exp(coords, bw)
            else:
                print 'Unsupported kernel function  ', kernel
        else:
            if kernel == 'gaussian':
            	self.wi = adapt_gauss(coords, bw)
            elif kernel == 'bisquare':
                self.wi = adapt_bisquare(coords, bw)
            elif kernel == 'exp':
                self.wi = adapt_exp(coords, bw)
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
        max_iter       : integer
                        Maximum number of iterations if convergence not
                        achieved
        solve         :string
                       Technique to solve MLE equations.
                       'iwls' = iteratively (re)weighted least squares (default)
        """
        self.fit_params['ini_betas'] = ini_betas
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve']= solve
        if solve.lower() == 'iwls':
            ey = self.y/self.offset
            if self.family=='Gaussian':
                results = GWRResults(self, *gauss_iwls(self))
            if self.family=='Poisson':
                results =  GWRResults(self, *poiss_iwls(self))
            if self.family=='logistic':
            	results = GWRResults(self, *logit_iwls(self))
        return results

class GWRResults(GWR):
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

        Methods: p215, (9.9), Fotheringham, Brunsdon and Charlton (2002)
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

        Methods: p215, (9.10), Fotheringham, Brunsdon and Charlton (2002)
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

        Methods: p55 (2.16)-(2.18), Fotheringham, Brunsdon and Charlton (2002),
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

    @property
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

    @property
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

    @property
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



