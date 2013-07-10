"""Probit regression class and diagnostics."""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import numpy.linalg as la
import scipy.optimize as op
from scipy.stats import norm, chisqprob
import scipy.sparse as SP
import user_output as USER
import summary_output as SUMMARY

__all__ = ["Probit"]


class BaseProbit: 
    """
    Probit class to do all the computations

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
                  Implemented as presented in Pinkse (2004)              
    KP_error    : float
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in Kelejian and Prucha (2001)
    PS_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in Pinkse and Slade (1998)
    warning     : boolean
                  if True Maximum number of iterations exceeded or gradient 
                  and/or function calls not changing.

    References
    ----------
    .. [1] Pinkse, J. (2004). Moran-flavored tests with nuisance parameter. In: Anselin,
    L., Florax, R. J., Rey, S. J. (editors) Advances in Spatial Econometrics,
    pages 67-77. Springer-Verlag, Heidelberg.
    .. [2] Kelejian, H., Prucha, I. (2001) "On the asymptotic distribution of the
    Moran I test statistic with applications". Journal of Econometrics, 104(2):219-57.
    .. [3] Pinkse, J., Slade, M. E. (1998) "Contracting in space: an application of
    spatial statistics to discrete-choice models". Journal of Econometrics, 85(1):125-54.

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
    def __init__(self,y,x,w=None,optim='newton',scalem='phimean',maxiter=100):
        self.y = y        
        self.x = x
        self.n, self.k = x.shape
        self.optim = optim
        self.scalem = scalem
        self.w = w
        self.maxiter = maxiter
        par_est, self.warning = self.par_est()
        self.betas = np.reshape(par_est[0],(self.k,1))
        self.logl = -float(par_est[1])
        self._cache = {}

    @property
    def vm(self):
        if 'vm' not in self._cache:
            H = self.hessian(self.betas)
            self._cache['vm'] = -la.inv(H)
        return self._cache['vm']
    @property
    def z_stat(self):
        if 'z_stat' not in self._cache:
            variance = self.vm.diagonal()
            zStat = self.betas.reshape(len(self.betas),)/ np.sqrt(variance)
            rs = {}
            for i in range(len(self.betas)):
                rs[i] = (zStat[i],norm.sf(abs(zStat[i]))*2)
            self._cache['z_stat'] = rs.values()
        return self._cache['z_stat']
    @property
    def slopes_std_err(self):
        if 'slopes_std_err' not in self._cache:
            variance = self.slopes_vm.diagonal() 
            self._cache['slopes_std_err'] = np.sqrt(variance)
        return self._cache['slopes_std_err'] 
    @property
    def slopes_z_stat(self):
        if 'slopes_z_stat' not in self._cache:
            zStat = self.slopes.reshape(len(self.slopes),)/self.slopes_std_err
            rs = {}
            for i in range(len(self.slopes)):
                rs[i] = (zStat[i],norm.sf(abs(zStat[i]))*2)
            self._cache['slopes_z_stat'] = rs.values()
        return self._cache['slopes_z_stat']    
    @property
    def xmean(self):
        if 'xmean' not in self._cache:
            self._cache['xmean'] = np.reshape(sum(self.x)/self.n,(self.k,1))
        return self._cache['xmean']
    @property
    def xb(self):
        if 'xb' not in self._cache:
            self._cache['xb'] = np.dot(self.x,self.betas)
        return self._cache['xb']    
    @property
    def predy(self):
        if 'predy' not in self._cache:
            self._cache['predy'] = norm.cdf(self.xb)
        return self._cache['predy']
    @property
    def predpc(self):
        if 'predpc' not in self._cache:
            predpc = abs(self.y-self.predy)
            for i in range(len(predpc)):
                if predpc[i]>0.5:
                    predpc[i]=0
                else:
                    predpc[i]=1
            self._cache['predpc'] = float(100* np.sum(predpc) / self.n)
        return self._cache['predpc']
    @property
    def phiy(self):
        if 'phiy' not in self._cache:
            self._cache['phiy'] = norm.pdf(self.xb)
        return self._cache['phiy']
    @property
    def scale(self):
        if 'scale' not in self._cache:
            if self.scalem == 'phimean':
                self._cache['scale'] = float(1.0 * np.sum(self.phiy)/self.n)
            if self.scalem == 'xmean':
                self._cache['scale'] = float(norm.pdf(np.dot(self.xmean.T,self.betas)))
        return self._cache['scale']
    @property
    def slopes(self):
        if 'slopes' not in self._cache:
            self._cache['slopes'] = self.betas[1:] * self.scale #Disregard the presence of dummies.
        return self._cache['slopes']
    @property
    def slopes_vm(self):
        if 'slopes_vm' not in self._cache:
            x = self.xmean
            b = self.betas
            dfdb = np.eye(self.k) - np.dot(b.T,x)*np.dot(b,x.T)
            slopes_vm = (self.scale**2)*np.dot(np.dot(dfdb,self.vm),dfdb.T)
            self._cache['slopes_vm'] = slopes_vm[1:,1:]
        return self._cache['slopes_vm']
    @property
    def LR(self):
        if 'LR' not in self._cache:    
            P = 1.0 * np.sum(self.y) / self.n
            LR = float(-2 * (self.n*(P * np.log(P) + (1 - P) * np.log(1 - P)) - self.logl))
            self._cache['LR'] = (LR,chisqprob(LR,self.k))
        return self._cache['LR']
    @property
    def u_naive(self):
        if 'u_naive' not in self._cache:
            u_naive = self.y - self.predy
            self._cache['u_naive'] = u_naive
        return self._cache['u_naive']
    @property
    def u_gen(self):
        if 'u_gen' not in self._cache:
            Phi_prod = self.predy * (1 - self.predy)
            u_gen = self.phiy * (self.u_naive / Phi_prod)
            self._cache['u_gen'] = u_gen
        return self._cache['u_gen']
    @property
    def Pinkse_error(self):
        if 'Pinkse_error' not in self._cache:
            self._cache['Pinkse_error'],self._cache['KP_error'],self._cache['PS_error'] = sp_tests(self)
        return self._cache['Pinkse_error']
    @property
    def KP_error(self):
        if 'KP_error' not in self._cache:
            self._cache['Pinkse_error'],self._cache['KP_error'],self._cache['PS_error'] = sp_tests(self)
        return self._cache['KP_error']
    @property
    def PS_error(self):
        if 'PS_error' not in self._cache:
            self._cache['Pinkse_error'],self._cache['KP_error'],self._cache['PS_error'] = sp_tests(self)
        return self._cache['PS_error']

    def par_est(self):
        start = np.dot(la.inv(np.dot(self.x.T,self.x)),np.dot(self.x.T,self.y))
        flogl = lambda par: -self.ll(par)
        if self.optim == 'newton':
            fgrad = lambda par: self.gradient(par)
            fhess = lambda par: self.hessian(par)            
            par_hat = newton(flogl,start,fgrad,fhess,self.maxiter)
            warn = par_hat[2]
        else:            
            fgrad = lambda par: -self.gradient(par)
            if self.optim == 'bfgs':
                par_hat = op.fmin_bfgs(flogl,start,fgrad,full_output=1,disp=0)
                warn = par_hat[6] 
            if self.optim == 'ncg':                
                fhess = lambda par: -self.hessian(par)
                par_hat = op.fmin_ncg(flogl,start,fgrad,fhess=fhess,full_output=1,disp=0)
                warn = par_hat[5]
        if warn > 0:
            warn = True
        else:
            warn = False
        return par_hat, warn

    def ll(self,par):       
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        ll = sum(np.log(norm.cdf(qxb)))
        return ll

    def gradient(self,par):      
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        qxb = q * np.dot(self.x,beta)
        lamb = q * norm.pdf(qxb)/norm.cdf(qxb)
        gradient = np.dot(lamb.T,self.x)[0]
        return gradient

    def hessian(self,par):           
        beta = np.reshape(np.array(par),(self.k,1))
        q = 2 * self.y - 1
        xb = np.dot(self.x,beta)
        qxb = q * xb
        lamb = q * norm.pdf(qxb)/norm.cdf(qxb)
        hessian = np.dot((self.x.T),(-lamb * (lamb + xb) * self.x ))
        return hessian

class Probit(BaseProbit): 
    """
    Classic non-spatial Probit and spatial diagnostics. The class includes a
    printout that formats all the results and tests in a nice format.

    The diagnostics for spatial dependence currently implemented are:

        * Pinkse Error [1]_
        * Kelejian and Prucha Moran's I [2]_
        * Pinkse & Slade Error [3]_

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
                  Implemented as presented in Pinkse (2004)              
    KP_error    : float
                  Moran's I type test against spatial error correlation.
                  Implemented as presented in Kelejian and Prucha (2001)
    PS_error    : float
                  Lagrange Multiplier test against spatial error correlation.
                  Implemented as presented in Pinkse and Slade (1998)
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
                   
    References
    ----------
    .. [1] Pinkse, J. (2004). Moran-flavored tests with nuisance parameter. In: Anselin, L., Florax, R. J., Rey, S. J. (editors) Advances in Spatial Econometrics, pages 67-77. Springer-Verlag, Heidelberg.
    .. [2] Kelejian, H., Prucha, I. (2001) "On the asymptotic distribution of the Moran I test statistic with applications". Journal of Econometrics, 104(2):219-57.
    .. [3] Pinkse, J., Slade, M. E. (1998) "Contracting in space: an application of spatial statistics to discrete-choice models". Journal of Econometrics, 85(1):125-54.

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
    CRIME variable into a binary variable. As in McMillen, D. (1992) "Probit with
    spatial autocorrelation". Journal of Regional Science 32(3):335-48, we define
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
    def __init__(self, y, x, w=None, optim='newton',scalem='phimean',maxiter=100,\
                 vm=False, name_y=None, name_x=None, name_w=None, name_ds=None, \
                 spat_diag=False):

        n = USER.check_arrays(y, x)
        USER.check_y(y, n)
        if w:
            USER.check_weights(w, y)
            spat_diag = True
        x_constant = USER.check_constant(x)
        BaseProbit.__init__(self,y=y,x=x_constant,w=w,optim=optim,scalem=scalem,maxiter=maxiter) 
        self.title = "CLASSIC PROBIT ESTIMATOR"        
        self.name_ds = USER.set_name_ds(name_ds)    
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_w = USER.set_name_w(name_w, w)
        SUMMARY.Probit(reg=self, w=w, vm=vm, spat_diag=spat_diag)    

def newton(flogl,start,fgrad,fhess,maxiter):
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
    while (iteration < maxiter and m>=1e-04):
        H = -la.inv(fhess(par_hat0))
        g = fgrad(par_hat0).reshape(start.shape)
        Hg = np.dot(H,g)
        par_hat0 = par_hat0 + Hg
        iteration += 1
        m = np.dot(g.T,Hg)
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
    if reg.w:
        w = reg.w.sparse
        Phi = reg.predy
        phi = reg.phiy                
        #Pinkse_error:
        Phi_prod = Phi * (1 - Phi)
        u_naive = reg.u_naive
        u_gen = reg.u_gen
        sig2 = np.sum((phi * phi) / Phi_prod) / reg.n
        LM_err_num = np.dot(u_gen.T,(w * u_gen))**2
        trWW = np.sum((w*w).diagonal())
        trWWWWp = trWW + np.sum((w*w.T).diagonal())
        LM_err = float(1.0 * LM_err_num / (sig2**2 * trWWWWp))
        LM_err = np.array([LM_err,chisqprob(LM_err,1)])
        #KP_error:
        moran = moran_KP(reg.w,u_naive,Phi_prod)
        #Pinkse-Slade_error:
        u_std = u_naive / np.sqrt(Phi_prod)
        ps_num = np.dot(u_std.T, (w * u_std))**2
        trWpW = np.sum((w.T*w).diagonal())
        ps = float(ps_num / (trWW + trWpW))
        ps = np.array([ps,chisqprob(ps,1)]) #chi-square instead of bootstrap.
    else:
        raise Exception, "W matrix not provided to calculate spatial test."
    return LM_err,moran,ps

def moran_KP(w,u,sig2i):
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
    w = w.sparse
    moran_num = np.dot(u.T, (w * u))
    E = SP.lil_matrix(w.get_shape())
    E.setdiag(sig2i.flat)
    E = E.asformat('csr')
    WE = w*E
    moran_den = np.sqrt(np.sum((WE*WE + (w.T*E)*WE).diagonal()))      
    moran = float(1.0*moran_num / moran_den)
    moran = np.array([moran,norm.sf(abs(moran)) * 2.])
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
    dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    y = np.array([dbf.by_col('CRIME')]).T
    var_x = ['INC', 'HOVAL']
    x = np.array([dbf.by_col(name) for name in var_x]).T    
    w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read()
    w.transform='r'
    probit1 = Probit((y>40).astype(float), x, w=w, name_x=var_x, name_y="CRIME",\
                     name_ds="Columbus", name_w="columbus.dbf")    
    #print probit1.summary
