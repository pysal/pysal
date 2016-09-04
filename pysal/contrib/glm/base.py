
from __future__ import print_function
import numpy as np
from scipy import stats
from utils import cache_readonly

class Results(object):
    """
    Class to contain model results
    Parameters
    ----------
    model : class instance
        the previously specified model instance
    params : array
        parameter estimates from the fit model
    """
    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)
        self._data_attr = []

    def initialize(self, model, params, **kwd):
        self.params = params
        self.model = model
        if hasattr(model, 'k_constant'):
            self.k_constant = model.k_constant

#TODO: public method?
class LikelihoodModelResults(Results):
    """
    Class to contain results from likelihood models
    Parameters
    -----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)
    Returns
    -------
    **Attributes**
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : array
        The t-values of the standard errors.
    Notes
    -----
    The covariance of params is given by scale times normalized_cov_params.
    Return values by solver if full_output is True during fit:
        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'lbfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                Warning flag:
                - 0 if converged
                - 1 if too many function evaluations or too many iterations
                - 2 if stopped for another reason
            converged : bool
                True: converged.  False: did not converge.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        """

    # by default we use normal distribution
    # can be overwritten by instances or subclasses
    use_t = False

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 **kwargs):
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale

        # robust covariance
        # We put cov_type in kwargs so subclasses can decide in fit whether to
        # use this generic implementation
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            if use_t is not None:
                self.use_t = use_t
        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})

            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description' : 'Standard Errors assume that the ' +
                                 'covariance matrix of the errors is correctly ' +
                                 'specified.'}
            else:
                from statsmodels.base.covtype import get_robustcov_results
                if cov_kwds is None:
                    cov_kwds = {}
                use_t = self.use_t
                # TODO: we shouldn't need use_t in get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                           use_t=use_t, **cov_kwds)


    def normalized_cov_params(self):
        raise NotImplementedError


    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True,
                                   use_t=None, **cov_kwds):
        from statsmodels.base.covtype import get_robustcov_results
        if cov_kwds is None:
            cov_kwds = {}

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description' : 'Standard Errors assume that the ' +
                             'covariance matrix of the errors is correctly ' +
                             'specified.'}
        else:
            # TODO: we shouldn't need use_t in get_robustcov_results
            get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                       use_t=use_t, **cov_kwds)

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse

    @cache_readonly
    def pvalues(self):
        if self.use_t:
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            return stats.t.sf(np.abs(self.tvalues), df_resid)*2
        else:
            return stats.norm.sf(np.abs(self.tvalues))*2


    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
            other=None):
        """
        Returns the variance/covariance matrix.
        The variance/covariance matrix can be of a linear contrast
        of the estimates of params or all params multiplied by scale which
        will usually be an estimate of sigma^2.  Scale is assumed to be
        a scalar.
        Parameters
        ----------
        r_matrix : array-like
            Can be 1d, or 2d.  Can be used alone or with other.
        column :  array-like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        other : array-like, optional
            Can be used when r_matrix is specified.
        Returns
        -------
        cov : ndarray
            covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.
        Notes
        -----
        (The below are assumed to be in matrix notation.)
        If no argument is specified returns the covariance matrix of a model
        ``(scale)*(X.T X)^(-1)``
        If contrast is specified it pre and post-multiplies as follows
        ``(scale) * r_matrix (X.T X)^(-1) r_matrix.T``
        If contrast and other are specified returns
        ``(scale) * r_matrix (X.T X)^(-1) other.T``
        If column is specified returns
        ``(scale) * (X.T X)^(-1)[column,column]`` if column is 0d
        OR
        ``(scale) * (X.T X)^(-1)[column][:,column]`` if column is 1d
        """
        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            dot_fun = nan_dot
        else:
            dot_fun = np.dot

        if (cov_p is None and self.normalized_cov_params is None and
            not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             '(unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other '
                             'arguments.')
        if other is not None and r_matrix is None:
            raise ValueError('other can only be specified with r_matrix')

        if cov_p is None:
            if hasattr(self, 'cov_params_default'):
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                #return cov_p[column][:, column]
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  # if r_matrix is None and column is None:
            return cov_p

    def conf_int(self, alpha=.05, cols=None, method='default'):
        """
        Returns the confidence interval of the fitted parameters.
        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return
        method : string
            Not Implemented Yet
            Method to estimate the confidence_interval.
            "Default" : uses self.bse which is based on inverse Hessian for MLE
            "hjjh" :
            "jac" :
            "boot-bse"
            "boot_quant"
            "profile"
        Returns
        --------
        conf_int : array
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.
        Examples
        --------
        >>> import pysal
        >>> from pysal.contrib.glm.glm import GLM
        >>> import numpy as np
        >>> db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
        >>> y = np.array(db.by_col("HOVAL")).reshape((-1,1))
        >>> X = []
        >>> X.append(db.by_col("INC"))
        >>> X.append(db.by_col("CRIME"))
        >>> X = np.array(X).T
        >>> model = GLM(y, X)
        >>> results = model.fit()
        >>> results.conf_int()
        array([[ 20.57281401,  72.28355135],
               [ -0.42138121,   1.67934915],
               [ -0.84292086,  -0.12685622]])

        Notes
        -----
        The confidence interval is based on the standard normal distribution.
        Models wish to use a different distribution should overwrite this
        method.
        """
        bse = self.bse

        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = self.params[cols] - q * bse[cols]
            upper = self.params[cols] + q * bse[cols]
        return np.asarray(lzip(lower, upper))

def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))
