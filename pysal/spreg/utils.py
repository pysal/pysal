"""
Tools for different procedure estimations
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, \
        Pedro V. Amaral pedro.amaral@asu.edu, \
        David C. Folch david.folch@asu.edu, \
        Daniel Arribas-Bel darribas@asu.edu"

import numpy as np
from scipy import sparse as SP
import scipy.optimize as op
import numpy.linalg as la
from pysal import lag_spatial
import copy


class RegressionPropsY:
    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    mean_y  : float
              Mean of the dependent variable
    std_y   : float
              Standard deviation of the dependent variable
              
    """

    @property
    def mean_y(self):
        if 'mean_y' not in self._cache:
            self._cache['mean_y']=np.mean(self.y)
        return self._cache['mean_y']
    @property
    def std_y(self):
        if 'std_y' not in self._cache:
            self._cache['std_y']=np.std(self.y, ddof=1)
        return self._cache['std_y']
    
class RegressionPropsVM:
    """
    Helper class that adds common regression properties to any regression
    class that inherits it.  It takes no parameters.  See BaseOLS for example
    usage.

    Parameters
    ----------

    Attributes
    ----------
    utu     : float
              Sum of the squared residuals
    sig2n    : float
              Sigma squared with n in the denominator
    sig2n_k : float
              Sigma squared with n-k in the denominator
    vm      : array
              Variance-covariance matrix (kxk)
              
    """

    @property
    def utu(self):
        if 'utu' not in self._cache:
            self._cache['utu'] = np.sum(self.u**2)
        return self._cache['utu']
    @property
    def sig2n(self):
        if 'sig2n' not in self._cache:
            self._cache['sig2n'] = self.utu / self.n
        return self._cache['sig2n']
    @property
    def sig2n_k(self):
        if 'sig2n_k' not in self._cache:
            self._cache['sig2n_k'] = self.utu / (self.n-self.k)
        return self._cache['sig2n_k']
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.xtxi)
        return self._cache['vm']    
    
def get_A1_het(S):
    """
    Builds A1 as in Arraiz et al [1]_

    .. math::

        A_1 = W' W - diag(w'_{.i} w_{.i})

    ...

    Parameters
    ----------

    S               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.
    """
    StS = S.T*S
    d = SP.spdiags([StS.diagonal()], [0], S.get_shape()[0], S.get_shape()[1])
    d = d.asformat('csr')
    return StS - d

def get_A1_hom(s, scalarKP=False):
    """
    Builds A1 for the spatial error GM estimation with homoscedasticity as in Drukker et al. [1]_ (p. 9).

    .. math::

        A_1 = \{1 + [n^{-1} tr(W'W)]^2\}^{-1} \[W'W - n^{-1} tr(W'W) I\]

    ...

    Parameters
    ----------

    s               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
    scalarKP        : boolean
                      Flag to include scalar corresponding to the first moment
                      condition as in Drukker et al. [1]_ (Defaults to False)

    Returns
    -------

    Implicit        : csr_matrix
                      A1 matrix in scipy sparse format
    References
    ----------

    .. [1] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.      
    """
    n = float(s.shape[0])
    wpw = s.T*s
    twpw = np.sum(wpw.diagonal()) 
    e = SP.eye(n, n, format='csr')
    e.data = np.ones(n) * (twpw / n)
    num = wpw - e
    if not scalarKP:
        return num
    else:
        den = 1. + (twpw / n)**2.
        return num / den

def get_A2_hom(s):
    """
    Builds A2 for the spatial error GM estimation with homoscedasticity as in
    Anselin (2011) [1]_ 

    .. math::

        A_2 = \dfrac{(W + W')}{2}

    ...

    Parameters
    ----------
    s               : csr_matrix
                      PySAL W object converted into Scipy sparse matrix
    Returns
    -------
    Implicit        : csr_matrix
                      A2 matrix in scipy sparse format
    References
    ----------

    .. [1] Anselin (2011) "GMM Estimation of Spatial Error Autocorrelation with and without Heteroskedasticity".
    """
    return (s + s.T) / 2.

def _moments2eqs(A1, s, u):
    '''
    Helper to compute G and g in a system of two equations as in
    the heteroskedastic error models from Drukker et al. [1]_
    ...

    Parameters
    ----------

    A1          : scipy.sparse.csr
                  A1 matrix as in the paper, different deppending on whether
                  it's homocedastic or heteroskedastic model

    s           : W.sparse
                  Sparse representation of spatial weights instance

    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.


    References
    ----------

    .. [1] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.
    '''
    n = float(s.shape[0])
    A1u = A1*u
    wu = s*u
    g1 = np.dot(u.T, A1u)
    g2 = np.dot(u.T, wu) 
    g = np.array([[g1][0][0],[g2][0][0]]) / n

    G11 = np.dot(u.T, ((A1 + A1.T)*wu))
    G12 = -np.dot((wu.T*A1), wu)
    G21 = np.dot(u.T, ((s + s.T)*wu))
    G22 = -np.dot(wu.T, (s*wu))
    G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / n
    return [G, g]

def optim_moments(moments_in, vcX=np.array([0])):
    """
    Optimization of moments
    ...

    Parameters
    ----------

    moments     : Moments
                  Instance of gmm_utils.moments_het with G and g
    vcX         : array
                  Optional. 2x2 array with the Variance-Covariance matrix to be used as
                  weights in the optimization (applies Cholesky
                  decomposition). Set empty by default.

    Returns
    -------
    x, f, d     : tuple
                  x -- position of the minimum
                  f -- value of func at the minimum
                  d -- dictionary of information from routine
                        d['warnflag'] is
                            0 if converged
                            1 if too many function evaluations
                            2 if stopped for another reason, given in d['task']
                        d['grad'] is the gradient at the minimum (should be 0 ish)
                        d['funcalls'] is the number of function calls made
    """
    moments = copy.deepcopy(moments_in)
    if vcX.any():
        Ec = np.transpose(la.cholesky(la.inv(vcX)))
        moments[0] = np.dot(Ec,moments_in[0])
        moments[1] = np.dot(Ec,moments_in[1])
    scale = np.min([[np.min(moments[0]),np.min(moments[1])]])
    moments[0],moments[1] = moments[0]/scale, moments[1]/scale
    if moments[0].shape[0] == 2:
        optim_par = lambda par: foptim_par(np.array([[float(par[0]),float(par[0])**2.]]).T,moments)
        start = [0.0]
        bounds=[(-1.0,1.0)]
    if moments[0].shape[0] == 3:
        optim_par = lambda par: foptim_par(np.array([[float(par[0]),float(par[0])**2.,float(par[1])]]).T,moments)
        start = [0.0,0.0]
        bounds=[(-1.0,1.0),(0.0,None)]        
    lambdaX = op.fmin_l_bfgs_b(optim_par,start,approx_grad=True,bounds=bounds)
    return lambdaX[0][0]

def foptim_par(par,moments):
    """ 
    Preparation of the function of moments for minimization
    ...

    Parameters
    ----------

    lambdapar       : float
                      Spatial autoregressive parameter
    moments         : list
                      List of Moments with G (moments[0]) and g (moments[1])

    Returns
    -------

    minimum         : float
                      sum of square residuals (e) of the equation system 
                      moments.g - moments.G * lambdapar = e
    """
    vv = np.dot(moments[0],par)
    vv2 = moments[1]-vv
    return sum(vv2**2)

def get_spFilter(w,lamb,sf):
    '''
    Compute the spatially filtered variables
    
    Parameters
    ----------
    w       : weight
              PySAL weights instance  
    lamb    : double
              spatial autoregressive parameter
    sf      : array
              the variable needed to compute the filter
    Returns
    --------
    rs      : array
              spatially filtered variable
    
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> w=pysal.open(pysal.examples.get_path("columbus.gal")).read()        
    >>> solu = get_spFilter(w,0.5,y)
    >>> print solu[0:5]
    [[  -8.9882875]
     [ -20.5685065]
     [ -28.196721 ]
     [ -36.9051915]
     [-111.1298   ]]

    '''
    try:
        result = sf - lamb * (w.sparse * sf)
    except:
        result = sf - lamb * (w * sf)
    return result

def get_lags(w, x, w_lags):
    '''
    Calculates a given order of spatial lags and all the smaller orders

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    x       : array
              nxk arrays with the variables to be lagged  
    w_lags  : integer
              Maximum order of spatial lag

    Returns
    --------
    rs      : array
              nxk*(w_lags+1) array with original and spatially lagged variables

    '''
    lag = lag_spatial(w, x)
    spat_lags = lag
    for i in range(w_lags-1):
        lag = lag_spatial(w, lag)
        spat_lags = sphstack(spat_lags, lag)
    return spat_lags

def inverse_prod(w, data, scalar, post_multiply=False, inv_method="power_exp", threshold=0.0000000001, max_iterations=None):
    """ 

    Parameters
    ----------

    w               : Pysal W object
                      nxn Pysal spatial weights object 

    data            : Numpy array
                      nx1 vector of data
    
    scalar          : float
                      Scalar value (typically rho or lambda)

    post_multiply   : boolean
                      If True then post-multiplies the data vector by the
                      inverse of the spatial filter, if false then
                      pre-multiplies.
    inv_method      : string
                      If "true_inv" uses the true inverse of W (slow);
                      If "power_exp" uses the power expansion method (default)

    threshold       : float
                      Test value to stop the iterations. Test is against
                      sqrt(increment' * increment), where increment is a
                      vector representing the contribution from each
                      iteration.

    max_iterations  : integer
                      Maximum number of iterations for the expansion.   

    Examples
    --------

    >>> import numpy, pysal
    >>> import numpy.linalg as la
    >>> np.random.seed(10)
    >>> w = pysal.lat2W(5, 5)
    >>> w.transform = 'r'
    >>> data = np.random.randn(w.n)
    >>> data.shape = (w.n, 1)
    >>> rho = 0.4
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp")
    >>> # true matrix inverse
    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv")
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True
    >>> # test the transpose version
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp", post_multiply=True)
    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv", post_multiply=True)
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True

    """                      
    if inv_method=="power_exp":
        inv_prod = power_expansion(w, data, scalar, post_multiply=post_multiply,\
                threshold=threshold, max_iterations=max_iterations)
    elif inv_method=="true_inv":
        try:
            matrix = la.inv(np.eye(w.n) - (scalar * w.full()[0]))
        except:
            matrix = la.inv(np.eye(w.shape[0]) - (scalar * w))
        if post_multiply:
            inv_prod = spdot(data.T, matrix)
        else:
            inv_prod = spdot(matrix, data)
    else:
        raise Exception, "Invalid method selected for inversion."
    return inv_prod

def power_expansion(w, data, scalar, post_multiply=False, threshold=0.0000000001, max_iterations=None):
    """
    Compute the inverse of a matrix using the power expansion (Leontief
    expansion).  General form is:
    
        .. math:: 
            x &= (I - \rho W)^{-1}v = [I + \rho W + \rho^2 WW + \dots]v \\
              &= v + \rho Wv + \rho^2 WWv + \dots

    Examples
    --------
    Tests for this function are in inverse_prod()

    """
    try:
        ws = w.sparse
    except:
        ws = w
    if post_multiply:
        data = data.T
    running_total = copy.copy(data)
    increment = copy.copy(data)
    count = 1
    test = 10000000
    if max_iterations == None:
        max_iterations = 10000000
    while test > threshold and count <= max_iterations:
        if post_multiply:    
            increment = increment*ws*scalar
        else:
            increment = ws*increment*scalar
        running_total += increment
        test_old = test
        test = la.norm(increment)
        if test > test_old:
            raise Exception, "power expansion will not converge, check model specification and that weight are less than 1"
        count += 1
    return running_total

def set_endog(y, x, w, yend, q, w_lags, lag_q):
    # Create spatial lag of y
    yl = lag_spatial(w, y)
    if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
        if lag_q:
            lag_vars = sphstack(x, q)
        else:
            lag_vars = x
        spatial_inst = get_lags(w ,lag_vars, w_lags)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)
    elif yend == None: # spatial instruments only
        q = get_lags(w, x, w_lags)
        yend = yl
    else:
        raise Exception, "invalid value passed to yend"
    return yend, q

    lag = lag_spatial(w, x)
    spat_lags = lag
    for i in range(w_lags-1):
        lag = lag_spatial(w, lag)
        spat_lags = sphstack(spat_lags, lag)
    return spat_lags


def set_endog_sparse(y, x, w, yend, q, w_lags, lag_q):
    """
    Same as set_endog, but with a sparse object passed as weights instead of W object.
    """
    yl = w * y
    if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
        if lag_q:
            lag_vars = sphstack(x, q)
        else:
            lag_vars = x
        spatial_inst = w * lag_vars
        for i in range(w_lags-1):
            spatial_inst = sphstack(spatial_inst, w * spatial_inst)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)
    elif yend == None: # spatial instruments only
        q = w * x
        for i in range(w_lags-1):
            q = sphstack(q, w * q)        
        yend = yl
    else:
        raise Exception, "invalid value passed to yend"
    return yend, q

def iter_msg(iteration,max_iter):
    if iteration==max_iter:
        iter_stop = "Maximum number of iterations reached."
    else:
        iter_stop = "Convergence threshold (epsilon) reached."
    return iter_stop

def sp_att(w,y,predy,w_y,rho):
    if np.abs(rho)<1:
        xb = predy - rho*w_y
        predy_sp = inverse_prod(w, xb, rho) 
        resid_sp = y - predy_sp
        return predy_sp, resid_sp
    else:
        return None, None

def spdot(a,b, array_out=True):
    """
    Matrix multiplication function to deal with sparse and dense objects

    Parameters
    ----------

    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------

    ab : array
         product of a times b. Sparse if a and b are sparse. Dense otherwise.
    """  
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = np.dot(a,b)
    elif type(a).__name__ == 'csr_matrix' or type(b).__name__ == 'csr_matrix' \
            or type(a).__name__ == 'csc_matrix' or type(b).__name__ == 'csc_matrix':
        ab = a*b
        if array_out:
            if type(ab).__name__ == 'csc_matrix' or type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception, "Invalid format for 'spdot' argument: %s and %s"%(type(a).__name__, type(b).__name__)
    return ab

def spmultiply(a, b, array_out=True):
    """
    Element-wise multiplication function to deal with sparse and dense
    objects. Both objects must be of the same type.

    Parameters
    ----------

    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
                  integer.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------

    ab : array
         elementwise multiplied object. Sparse if a is sparse. Dense otherwise.
    """  
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = a*b
    elif (type(a).__name__ == 'csr_matrix' or type(a).__name__ == 'csc_matrix') \
         and (type(b).__name__ == 'csr_matrix' or type(b).__name__ == 'csc_matrix'):
        ab = a.multiply(b)
        if array_out:
            if type(ab).__name__ == 'csc_matrix' or type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception, "Invalid format for 'spmultiply' argument: %s and %s"%(type(a).__name__, type(b).__name__)
    return ab

def sphstack(a,b, array_out=False):
    """
    Horizontal stacking of vectors (or matrices) to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  First object.
    b           : array or sparse matrix
                  Object to be stacked next to a
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------

    ab          : array or sparse matrix
                  Horizontally stacked objects
    """  
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = np.hstack((a,b))
    elif type(a).__name__ == 'csr_matrix' or type(b).__name__ == 'csr_matrix':
        ab = SP.hstack((a,b), format='csr')
        if array_out:
            if type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception, "Invalid format for 'sphstack' argument: %s and %s"%(type(a).__name__, type(b).__name__)
    return ab

def spbroadcast(a,b, array_out=False):
    """
    Element-wise multiplication of a matrix and vector to deal with sparse 
    and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.
    b           : array
                  Object with only one column
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------

    ab          : array or sparse matrix
                  Element-wise multiplication of a and b
    """  
    if type(a).__name__ == 'ndarray' and type(b).__name__ == 'ndarray':
        ab = a*b
    elif type(a).__name__ == 'csr_matrix':
        b_mod = SP.lil_matrix((b.shape[0], b.shape[0]))
        b_mod.setdiag(b)
        ab = (a.T*b_mod).T
        if array_out:
            if type(ab).__name__ == 'csr_matrix':
                ab = ab.toarray()
    else:
        raise Exception, "Invalid format for 'spbroadcast' argument: %s and %s"%(type(a).__name__, type(b).__name__)
    return ab

def spmin(a):
    """
    Minimum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------

    min a       : int or float
                  minimum value in a
    """  


    if type(a).__name__ == 'ndarray':
        return a.min()
    elif type(a).__name__ == 'csr_matrix' or type(a).__name__ == 'csc_matrix':
        try:
            return min(a.data)
        except:
            if np.sum(a.data) == 0:
                return 0
            else:
                raise Exception, "Error: could not evaluate the minimum value."
    else:
        raise Exception, "Invalid format for 'spmultiply' argument: %s and %s"%(type(a).__name__, type(b).__name__)

def spmax(a):
    """
    Maximum value in a matrix or vector to deal with sparse and dense objects

    Parameters
    ----------

    a           : array or sparse matrix
                  Object with one or more columns.

    Returns
    -------

    max a       : int or float
                  maximum value in a
    """  
    if type(a).__name__ == 'ndarray':
        return a.max()
    elif type(a).__name__ == 'csr_matrix' or type(a).__name__ == 'csc_matrix':
        try:
            return max(a.data)
        except:
            if np.sum(a.data) == 0:
                return 0
            else:
                raise Exception, "Error: could not evaluate the maximum value."    
    else:
        raise Exception, "Invalid format for 'spmultiply' argument: %s and %s"%(type(a).__name__, type(b).__name__)

def set_warn(reg,warn):
    if warn:
        try:
            reg.warning += warn
        except:
            reg.warning = warn
    else:
        reg.warning = None

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test() 
