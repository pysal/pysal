"""Probit regression class and diagnostics."""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import numpy.linalg as la
import scipy.optimize as op
from scipy.stats import norm, chisqprob
import scipy.sparse as SP
import user_output as USER
import summary_output as SUMMARY
from utils import spdot, spbroadcast

__all__ = ["Probit"]


class BaseProbit(object):

    """
    Probit class to do all the computations

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent binary variable
    w           : W
                  PySAL weights instance or spatial weights sparse matrix
                  aligned with y
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternatives: 'ncg' (Newton-CG), 'bfgs' (BFGS algorithm)
    scalem      : string
                  Method to calculate the scale of the marginal effects.
                  Default: 'phimean' (Mean of individual marginal effects)
                  Alternative: 'xmean' (Marginal effects at variables mean)
    maxiter     : int
                  Maximum number of iterations until optimizer stops                  

    Attributes
    ----------

    x           : array
                  Two dimensional array with n rows and one column for each
                  independent (exogenous) variable, including the constant
    y           : array
                  nx1 array of dependent variable
    betas       : array
                  kx1 array with estimated coefficients
    predy       : array
                  nx1 array of predicted y values
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    vm          : array
                  Variance-covariance matrix (kxk)
    z_stat      : list of tuples
                  z statistic; each tuple contains the pair (statistic,
                  p-value), where each is a float                  
    xmean       : array
                  Mean of the independent variables (kx1)
    predpc      : float
                  Percent of y correctly predicted
    logl        : float
                  Log-Likelihhod of the estimation
    scalem      : string
                  Method to calculate the scale of the marginal effects.
    scale       : float
                  Scale of the marginal effects.
    slopes      : array
                  Marginal effects of the independent variables (k-1x1)
                  
		  Note: Disregards the presence of dummies.
    slopes_vm   : array
                  Variance-covariance matrix of the slopes (k-1xk-1)
    LR          : tuple
                  Likelihood Ratio test of all coefficients = 0
                  
		  (test statistics, p-value)
    Pinkse_error: float
                  Lagrange Multiplier test against spatial error correlation.
                  
		  Implemented as presented in [Pinkse2004]_              
    KP_error    : float
                  Moran's I type test against spatial error correlation.
                  
		  Implemented as presented in  [Kelejian2001]_
    PS_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in  [Pinkse1998]_
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient 
                  and/or function calls not changing.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('HOVAL')]).T
    >>> x = np.hstack((np.ones(y.shape),x))
    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    >>> w.transform='r'
    >>> model = BaseProbit((y>40).astype(float), x, w=w)    
    >>> np.around(model.betas, decimals=6)
    array([[ 3.353811],
           [-0.199653],
           [-0.029514]])

    >>> np.around(model.vm, decimals=6)
    array([[ 0.852814, -0.043627, -0.008052],
           [-0.043627,  0.004114, -0.000193],
           [-0.008052, -0.000193,  0.00031 ]])

    >>> tests = np.array([['Pinkse_error','KP_error','PS_error']])
    >>> stats = np.array([[model.Pinkse_error[0],model.KP_error[0],model.PS_error[0]]])
    >>> pvalue = np.array([[model.Pinkse_error[1],model.KP_error[1],model.PS_error[1]]])
    >>> print np.hstack((tests.T,np.around(np.hstack((stats.T,pvalue.T)),6)))
    [['Pinkse_error' '3.131719' '0.076783']
     ['KP_error' '1.721312' '0.085194']
     ['PS_error' '2.558166' '0.109726']]
    """

    def __init__(self, y, x, w=None, optim='newton', scalem='phimean', maxiter=100):
        self.y = y
        self.x = x
        self.n, self.k = x.shape
        self.optim = optim
        self.scalem = scalem
        self.w = w
        self.maxiter = maxiter
        par_est, self.warning = self.par_est()
        self.betas = np.reshape(par_est[0], (self.k, 1))
        self.logl = -float(par_est[1])

    @property
    def vm(self):
        try:
            return self._cache['vm']
        except AttributeError:
            self._cache = {}
            H = self.hessian(self.betas)
            self._cache['vm'] = -la.inv(H)
        except KeyError:
            H = self.hessian(self.betas)
            self._cache['vm'] = -la.inv(H)
        return self._cache['vm']
    
    @vm.setter
    def vm(self, val):
        try:
            self._cache['vm'] = val
        except AttributeError:
            self._cache = {}
        self._cache['vm'] = val

    @property #could this get packaged into a separate function or something? It feels weird to duplicate this.  
    def z_stat(self):
        try: 
            return self._cache['z_stat']
        except AttributeError:
            self._cache = {}
            variance = self.vm.diagonal()
            zStat = self.betas.reshape(len(self.betas),) / np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i], norm.sf(abs(zStat[i])) * 2)
            self._cache['z_stat'] = rs.values()
        except KeyError:
            variance = self.vm.diagonal()
            zStat = self.betas.reshape(len(self.betas),) / np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i], norm.sf(abs(zStat[i])) * 2)
            self._cache['z_stat'] = rs.values()
        return self._cache['z_stat']

    @z_stat.setter
    def z_stat(self, val):
        try:
            self._cache['z_stat'] = val
        except AttributeError:
            self._cache = {}
        self._cache['z_stat'] = val

    @property
    def slopes_std_err(self):
        try:
            return self._cache['slopes_std_err']
        except AttributeError:
            self._cache = {}
            self._cache['slopes_std_err'] = np.sqrt(self.slopes_vm.diagonal())
        except KeyError:
            self._cache['slopes_std_err'] = np.sqrt(self.slopes_vm.diagonal())
        return self._cache['slopes_std_err']    
    
    @slopes_std_err.setter
    def slopes_std_err(self, val):
        try:
            self._cache['slopes_std_err'] = val
        except AttributeError:
            self._cache = {}
        self._cache['slopes_std_err'] = val

    @property
    def slopes_z_stat(self):
        try:
            return self._cache['slopes_z_stat']
        except AttributeError:
            self._cache = {}
            zStat = self.slopes.reshape(
                len(self.slopes),) / self.slopes_std_err
            rs = {}
            for i in range(len(self.slopes)):
                rs[i] = (zStat[i], norm.sf(abs(zStat[i])) * 2)
            self._cache['slopes_z_stat'] = rs.values()
        except KeyError:
            zStat = self.slopes.reshape(
                len(self.slopes),) / self.slopes_std_err
            rs = {}
            for i in range(len(self.slopes)):
                rs[i] = (zStat[i], norm.sf(abs(zStat[i])) * 2)
            self._cache['slopes_z_stat'] = rs.values()
        return self._cache['slopes_z_stat']

    @slopes_z_stat.setter
    def slopes_z_stat(self, val):
        try:
            self._cache['slopes_z_stat'] = val
        except AttributeError:
            self._cache = {}
        self._cache['slopes_z_stat'] = val

    @property
    def xmean(self):
        try:
            return self._cache['xmean']
        except AttributeError:
            self._cache = {}
            try: #why is this try-accept? can x be a list??
                self._cache['xmean'] = np.reshape(sum(self.x) / self.n, (self.k, 1))
            except:
                self._cache['xmean'] = np.reshape(sum(self.x).toarray() / self.n, (self.k, 1))
        except KeyError:
            try:
                self._cache['xmean'] = np.reshape(sum(self.x) / self.n, (self.k, 1))
            except:
                self._cache['xmean'] = np.reshape(sum(self.x).toarray() / self.n, (self.k, 1))
        return self._cache['xmean']

    @xmean.setter
    def xmean(self, val):
        try:
            self._cache['xmean'] = val
        except AttributeError:
            self._cache = {}
        self._cache['xmean'] = val

    @property
    def xb(self):
        try:
            return self._cache['xb']
        except AttributeError:
            self._cache = {}
            self._cache['xb'] = spdot(self.x, self.betas)
        except KeyError:
            self._cache['xb'] = spdot(self.x, self.betas)
        return self._cache['xb']

    @xb.setter
    def xb(self, val):
        try:
            self._cache['xb'] = val
        except AttributeError:
            self._cache = {}
        self._cache['xb'] = val

    @property
    def predy(self):
        try:
            return self._cache['predy']
        except AttributeError:
            self._cache = {}
            self._cache['predy'] = norm.cdf(self.xb)
        except KeyError:
            self._cache['predy'] = norm.cdf(self.xb)
        return self._cache['predy']
    
    @predy.setter
    def predy(self, val):
        try:
            self._cache['predy'] = val
        except AttributeError:
            self._cache = {}
        self._cache['predy'] = val

    @property
    def predpc(self):
        try:
            return self._cache['predpc']
        except AttributeError:
            self._cache = {}
            predpc = abs(self.y - self.predy)
            for i in range(len(predpc)):
                if predpc[i] > 0.5:
                    predpc[i] = 0
                else:
                    predpc[i] = 1
            self._cache['predpc'] = float(100.0 * np.sum(predpc) / self.n)
        except KeyError:
            predpc = abs(self.y - self.predy)
            for i in range(len(predpc)):
                if predpc[i] > 0.5:
                    predpc[i] = 0
                else:
                    predpc[i] = 1
            self._cache['predpc'] = float(100.0 * np.sum(predpc) / self.n)
        return self._cache['predpc']

    @predpc.setter
    def predpc(self, val):
        try:
            self._cache['predpc'] = val
        except AttributeError:
            self._cache = {}
        self._cache['predpc'] = val
    
    @property
    def phiy(self):
        try:
            return self._cache['phiy']
        except AttributeError:
            self._cache = {}
            self._cache['phiy'] = norm.pdf(self.xb)
        except KeyError:
            self._cache['phiy'] = norm.pdf(self.xb)
        return self._cache['phiy']
    
    @phiy.setter
    def phiy(self, val):
        try:
            self._cache['phiy'] = val
        except AttributeError:
            self._cache = {}
        self._cache['phiy'] = val

    @property
    def scale(self):
        try:
            return self._cache['scale']
        except AttributeError:
            self._cache = {}
            if self.scalem == 'phimean':
                self._cache['scale'] = float(1.0 * np.sum(self.phiy) / self.n)
            elif self.scalem == 'xmean':
                self._cache['scale'] = float(norm.pdf(np.dot(self.xmean.T, self.betas)))
        except KeyError:
            if self.scalem == 'phimean':
                self._cache['scale'] = float(1.0 * np.sum(self.phiy) / self.n)
            if self.scalem == 'xmean':
                self._cache['scale'] = float(norm.pdf(np.dot(self.xmean.T, self.betas)))
        return self._cache['scale']

    @scale.setter
    def scale(self, val):
        try:
            self._cache['scale'] = val
        except AttributeError:
            self._cache = {}
        self._cache['scale'] = val

    @property
    def slopes(self):
        try:
            return self._cache['slopes']
        except AttributeError:
            self._cache = {}
            self._cache['slopes'] = self.betas[1:] * self.scale
        except KeyError:
            self._cache['slopes'] = self.betas[1:] * self.scale
        return self._cache['slopes']

    @slopes.setter
    def slopes(self, val):
        try:
            self._cache['slopes'] = val
        except AttributeError:
            self._cache = {}
        self._cache['slopes'] = val

    @property
    def slopes_vm(self):
        try:
            return self._cache['slopes_vm']
        except AttributeError:
            self._cache = {}
            x = self.xmean
            b = self.betas
            dfdb = np.eye(self.k) - spdot(b.T, x) * spdot(b, x.T)
            slopes_vm = (self.scale ** 2) * \
                np.dot(np.dot(dfdb, self.vm), dfdb.T)
            self._cache['slopes_vm'] = slopes_vm[1:, 1:]
        except KeyError:
            x = self.xmean
            b = self.betas
            dfdb = np.eye(self.k) - spdot(b.T, x) * spdot(b, x.T)
            slopes_vm = (self.scale ** 2) * \
                np.dot(np.dot(dfdb, self.vm), dfdb.T)
            self._cache['slopes_vm'] = slopes_vm[1:, 1:]
        return self._cache['slopes_vm']

    @slopes_vm.setter
    def slopes_vm(self, val):
        try:
            self._cache['slopes_vm'] = val
        except AttributeError:
            self._cache = {}
        self._cache['slopes_vm'] = val

    @property
    def LR(self):
        try:
            return self._cache['LR']
        except AttributeError:
            self._cache = {}
            P = 1.0 * np.sum(self.y) / self.n
            LR = float(
                -2 * (self.n * (P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logl))
            self._cache['LR'] = (LR, chisqprob(LR, self.k))
        except KeyError:
            P = 1.0 * np.sum(self.y) / self.n
            LR = float(
                -2 * (self.n * (P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logl))
            self._cache['LR'] = (LR, chisqprob(LR, self.k))
        return self._cache['LR']

    @LR.setter
    def LR(self, val):
        try:
            self._cache['LR'] = val
        except AttributeError:
            self._cache = {}
        self._cache['LR'] = val

    @property
    def u_naive(self):
        try:
            return self._cache['u_naive']
        except AttributeError:
            self._cache = {}
            self._cache['u_naive'] = self.y - self.predy
        except KeyError:
            u_naive = self.y - self.predy
            self._cache['u_naive'] = u_naive
        return self._cache['u_naive']

    @u_naive.setter
    def u_naive(self, val):
        try:
            self._cache['u_naive'] = val
        except AttributeError:
            self._cache = {}
        self._cache['u_naive'] = val
    
    @property
    def u_gen(self):
        try:
            return self._cache['u_gen']
        except AttributeError:
            self._cache = {}
            Phi_prod = self.predy * (1 - self.predy)
            u_gen = self.phiy * (self.u_naive / Phi_prod)
            self._cache['u_gen'] = u_gen
        except KeyError:
            Phi_prod = self.predy * (1 - self.predy)
            u_gen = self.phiy * (self.u_naive / Phi_prod)
            self._cache['u_gen'] = u_gen
        return self._cache['u_gen']
    
    @u_gen.setter
    def u_gen(self, val):
        try:
            self._cache['u_gen'] = val
        except AttributeError:
            self._cache = {}
        self._cache['u_gen'] = val

    @property
    def Pinkse_error(self):
        try:
            return self._cache['Pinkse_error']
        except AttributeError:
            self._cache = {}
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        except KeyError:
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        return self._cache['Pinkse_error']

    @Pinkse_error.setter
    def Pinkse_error(self, val):
        try:
            self._cache['Pinkse_error'] = val
        except AttributeError:
            self._cache = {}
        self._cache['Pinkse_error'] = val

    @property
    def KP_error(self):
        try:
            return self._cache['KP_error']
        except AttributeError:
            self._cache = {}
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        except KeyError:
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        return self._cache['KP_error']

    @KP_error.setter
    def KP_error(self, val):
        try:
            self._cache['KP_error'] = val
        except AttributeError:
            self._cache = {}
        self._cache['KP_error'] = val

    @property
    def PS_error(self):
        try:
            return self._cache['PS_error']
        except AttributeError:
            self._cache = {}
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        except KeyError:
            self._cache['Pinkse_error'], self._cache[
                'KP_error'], self._cache['PS_error'] = sp_tests(self)
        return self._cache['PS_error']

    @PS_error.setter
    def PS_error(self, val):
        try:
            self._cache['PS_error'] = val
        except AttributeError:
            self._cache = {}
        self._cache['PS_error'] = val

    def par_est(self):
        start = np.dot(la.inv(spdot(self.x.T, self.x)),
                       spdot(self.x.T, self.y))
        flogl = lambda par: -self.ll(par)
        if self.optim == 'newton':
            fgrad = lambda par: self.gradient(par)
            fhess = lambda par: self.hessian(par)
            par_hat = newton(flogl, start, fgrad, fhess, self.maxiter)
            warn = par_hat[2]
        else:
            fgrad = lambda par: -self.gradient(par)
            if self.optim == 'bfgs':
                par_hat = op.fmin_bfgs(
                    flogl, start, fgrad, full_output=1, disp=0)
                warn = par_hat[6]
            if self.optim == 'ncg':
                fhess = lambda par: -self.hessian(par)
                par_hat = op.fmin_ncg(
                    flogl, start, fgrad, fhess=fhess, full_output=1, disp=0)
                warn = par_hat[5]
        if warn > 0:
            warn = True
        else:
            warn = False
        return par_hat, warn

    def ll(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        qxb = q * spdot(self.x, beta)
        ll = sum(np.log(norm.cdf(qxb)))
        return ll

    def gradient(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        qxb = q * spdot(self.x, beta)
        lamb = q * norm.pdf(qxb) / norm.cdf(qxb)
        gradient = spdot(lamb.T, self.x)[0]
        return gradient

    def hessian(self, par):
        beta = np.reshape(np.array(par), (self.k, 1))
        q = 2 * self.y - 1
        xb = spdot(self.x, beta)
        qxb = q * xb
        lamb = q * norm.pdf(qxb) / norm.cdf(qxb)
        hessian = spdot(self.x.T, spbroadcast(self.x,-lamb * (lamb + xb)))
        return hessian


class Probit(BaseProbit):

    """
    Classic non-spatial Probit and spatial diagnostics. The class includes a
    printout that formats all the results and tests in a nice format.

    The diagnostics for spatial dependence currently implemented are:

        * Pinkse Error [Pinkse2004]_
        * Kelejian and Prucha Moran's I [Kelejian2001]_
        * Pinkse & Slade Error [Pinkse1998]_

    Parameters
    ----------

    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent binary variable
    w           : W
                  PySAL weights instance aligned with y
    optim       : string
                  Optimization method.
                  Default: 'newton' (Newton-Raphson).
                  Alternatives: 'ncg' (Newton-CG), 'bfgs' (BFGS algorithm)
    scalem      : string
                  Method to calculate the scale of the marginal effects.
                  Default: 'phimean' (Mean of individual marginal effects)
                  Alternative: 'xmean' (Marginal effects at variables mean)
    maxiter     : int
                  Maximum number of iterations until optimizer stops                  
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Attributes
    ----------

    x           : array
                  Two dimensional array with n rows and one column for each
                  independent (exogenous) variable, including the constant
    y           : array
                  nx1 array of dependent variable
    betas       : array
                  kx1 array with estimated coefficients
    predy       : array
                  nx1 array of predicted y values
    n           : int
                  Number of observations
    k           : int
                  Number of variables
    vm          : array
                  Variance-covariance matrix (kxk)
    z_stat      : list of tuples
                  z statistic; each tuple contains the pair (statistic,
                  p-value), where each is a float                  
    xmean       : array
                  Mean of the independent variables (kx1)
    predpc      : float
                  Percent of y correctly predicted
    logl        : float
                  Log-Likelihhod of the estimation
    scalem      : string
                  Method to calculate the scale of the marginal effects.
    scale       : float
                  Scale of the marginal effects.
    slopes      : array
                  Marginal effects of the independent variables (k-1x1)
    slopes_vm   : array
                  Variance-covariance matrix of the slopes (k-1xk-1)
    LR          : tuple
                  Likelihood Ratio test of all coefficients = 0
                  (test statistics, p-value)
    Pinkse_error: float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in  [Pinkse2004]_             
    KP_error    : float
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in [Kelejian2001]_
    PS_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in [Pinkse1998]_
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient 
                  and/or function calls not changing.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')

    Extract the CRIME column (crime) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept. Since we want to run a probit model and for this
    example we use the Columbus data, we also need to transform the continuous
    CRIME variable into a binary variable. As in [McMillen1992]_, we define
    y = 1 if CRIME > 40.

    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> y = (y>40).astype(float)

    Extract HOVAL (home values) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> names_to_extract = ['INC', 'HOVAL']
    >>> x = np.array([dbf.by_col(name) for name in names_to_extract]).T

    Since we want to the test the probit model for spatial dependence, we need to
    specify the spatial weights matrix that includes the spatial configuration of
    the observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. In PySAL, this
    can be easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional. 

    >>> model = Probit(y, x, w=w, name_y='crime', name_x=['income','home value'], name_ds='columbus', name_w='columbus.gal')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them.

    >>> np.around(model.betas, decimals=6)
    array([[ 3.353811],
           [-0.199653],
           [-0.029514]])

    >>> np.around(model.vm, decimals=6)
    array([[ 0.852814, -0.043627, -0.008052],
           [-0.043627,  0.004114, -0.000193],
           [-0.008052, -0.000193,  0.00031 ]])

    Since we have provided a spatial weigths matrix, the diagnostics for
    spatial dependence have also been computed. We can access them and their
    p-values individually:

    >>> tests = np.array([['Pinkse_error','KP_error','PS_error']])
    >>> stats = np.array([[model.Pinkse_error[0],model.KP_error[0],model.PS_error[0]]])
    >>> pvalue = np.array([[model.Pinkse_error[1],model.KP_error[1],model.PS_error[1]]])
    >>> print np.hstack((tests.T,np.around(np.hstack((stats.T,pvalue.T)),6)))
    [['Pinkse_error' '3.131719' '0.076783']
     ['KP_error' '1.721312' '0.085194']
     ['PS_error' '2.558166' '0.109726']]

    Or we can easily obtain a full summary of all the results nicely formatted and
    ready to be printed simply by typing 'print model.summary'

    """

    def __init__(
        self, y, x, w=None, optim='newton', scalem='phimean', maxiter=100,
        vm=False, name_y=None, name_x=None, name_w=None, name_ds=None,
            spat_diag=False):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        if w != None:
            USER.check_weights(w, y)
            spat_diag = True
            ws = w.sparse
        else:
            ws = None
        x_constant = USER.check_constant(x)
        BaseProbit.__init__(self, y=y, x=x_constant, w=ws,
                            optim=optim, scalem=scalem, maxiter=maxiter)
        self.title = "CLASSIC PROBIT ESTIMATOR"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.Probit(reg=self, w=w, vm=vm, spat_diag=spat_diag)


def newton(flogl, start, fgrad, fhess, maxiter):
    """
    Calculates the Newton-Raphson method

    Parameters
    ----------

    flogl       : lambda
                  Function to calculate the log-likelihood
    start       : array
                  kx1 array of starting values
    fgrad       : lambda
                  Function to calculate the gradient
    fhess       : lambda
                  Function to calculate the hessian
    maxiter     : int
                  Maximum number of iterations until optimizer stops                
    """
    warn = 0
    iteration = 0
    par_hat0 = start
    m = 1
    while (iteration < maxiter and m >= 1e-04):
        H = -la.inv(fhess(par_hat0))
        g = fgrad(par_hat0).reshape(start.shape)
        Hg = np.dot(H, g)
        par_hat0 = par_hat0 + Hg
        iteration += 1
        m = np.dot(g.T, Hg)
    if iteration == maxiter:
        warn = 1
    logl = flogl(par_hat0)
    return (par_hat0, logl, warn)


def sp_tests(reg):
    """
    Calculates tests for spatial dependence in Probit models

    Parameters
    ----------

    reg         : regression object
                  output instance from a probit model            
    """
    if reg.w != None:
        try:
            w = reg.w.sparse
        except:
            w = reg.w
        Phi = reg.predy
        phi = reg.phiy
        # Pinkse_error:
        Phi_prod = Phi * (1 - Phi)
        u_naive = reg.u_naive
        u_gen = reg.u_gen
        sig2 = np.sum((phi * phi) / Phi_prod) / reg.n
        LM_err_num = np.dot(u_gen.T, (w * u_gen)) ** 2
        trWW = np.sum((w * w).diagonal())
        trWWWWp = trWW + np.sum((w * w.T).diagonal())
        LM_err = float(1.0 * LM_err_num / (sig2 ** 2 * trWWWWp))
        LM_err = np.array([LM_err, chisqprob(LM_err, 1)])
        # KP_error:
        moran = moran_KP(reg.w, u_naive, Phi_prod)
        # Pinkse-Slade_error:
        u_std = u_naive / np.sqrt(Phi_prod)
        ps_num = np.dot(u_std.T, (w * u_std)) ** 2
        trWpW = np.sum((w.T * w).diagonal())
        ps = float(ps_num / (trWW + trWpW))
        # chi-square instead of bootstrap.
        ps = np.array([ps, chisqprob(ps, 1)])
    else:
        raise Exception, "W matrix must be provided to calculate spatial tests."
    return LM_err, moran, ps


def moran_KP(w, u, sig2i):
    """
    Calculates Moran-flavoured tests 

    Parameters
    ----------

    w           : W
                  PySAL weights instance aligned with y
    u           : array
                  nx1 array of naive residuals
    sig2i       : array
                  nx1 array of individual variance               
    """
    try:
        w = w.sparse
    except:
        pass
    moran_num = np.dot(u.T, (w * u))
    E = SP.lil_matrix(w.get_shape())
    E.setdiag(sig2i.flat)
    E = E.asformat('csr')
    WE = w * E
    moran_den = np.sqrt(np.sum((WE * WE + (w.T * E) * WE).diagonal()))
    moran = float(1.0 * moran_num / moran_den)
    moran = np.array([moran, norm.sf(abs(moran)) * 2.])
    return moran


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    dbf = pysal.open(pysal.examples.get_path('columbus.dbf'), 'r')
    y = np.array([dbf.by_col('CRIME')]).T
    var_x = ['INC', 'HOVAL']
    x = np.array([dbf.by_col(name) for name in var_x]).T
    w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    w.transform = 'r'
    probit1 = Probit(
        (y > 40).astype(float), x, w=w, name_x=var_x, name_y="CRIME",
        name_ds="Columbus", name_w="columbus.dbf")
    print probit1.summary
