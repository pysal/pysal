from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import copy

from ...both_levels.generic import Base_Generic
from ...both_levels.generic.model import SAMPLERS as generic_parameters
from ... import verify
from ...utils import sma_covariance, sma_precision, ind_covariance, chol_mvn


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2', 'Lambda']

class Base_Upper_SMA(Base_Generic):
    """
    All arguments are documented in Upper_SMA
    """
    def __init__(self, Y, X, M, Delta, n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None,
                 configs=None,
                 truncation=None):
        W = np.eye((Delta.shape[0]))
        super(Base_Upper_SMA, self).__init__(Y, X, W, M, Delta,
                                            n_samples=0, n_jobs=n_jobs,
                                            extra_traced_params=None,
                                            priors=priors,
                                            configs=configs,
                                            starting_values=starting_values,
                                            truncation=truncation)
        st = self.state
        self.state.Psi_1 = ind_covariance
        self.state.Psi_1i = ind_covariance
        self.state.Psi_2 = sma_covariance
        self.state.Psi_2i = sma_precision
        original_traced = copy.deepcopy(self.traced_params)
        to_drop = [k for k in original_traced if (k not in SAMPLERS and k in generic_parameters)]
        self.traced_params = copy.deepcopy(SAMPLERS)
        for param in to_drop:
            for i, _  in enumerate(self.trace.chains):
                del self.trace.chains[i][param]
        extra_traced_params = [] if extra_traced_params is None else extra_traced_params
        for extra in extra_traced_params:
            self.traced_params.append(extra)
            for i, chain in enumerate(self.trace.chains):
                chain[extra] = []


        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _iteration(self):
        st = self.state

        ### Sample the Beta conditional posterior
        ### P(beta | . ) \propto L(Y|.) \dot P(\beta)
        ### is
        ### N(Sb, S) where
        ### S = (X' Sigma^{-1}_Y X + S_0^{-1})^{-1}
        ### b = X' Sigma^{-1}_Y (Y - Delta Alphas) + S^{-1}\mu_0
        covm_update = st.X.T.dot(st.X) / st.Sigma2
        covm_update += st.Betas_cov0i
        covm_update = la.inv(covm_update)

        resids = st.Y - st.Delta.dot(st.Alphas)
        XtSresids = st.X.T.dot(resids) / st.Sigma2
        mean_update = XtSresids + st.Betas_cov0i.dot(st.Betas_mean0)
        mean_update = np.dot(covm_update, mean_update)
        st.Betas = chol_mvn(mean_update, covm_update)
        st.XBetas = np.dot(st.X, st.Betas)

        ### Sample the Random Effect conditional posterior
        ### P( Alpha | . ) \propto L(Y|.) \dot P(Alpha | \lambda, Tau2)
        ###                               \dot P(Tau2) \dot P(\lambda)
        ### is
        ### N(Sb, S)
        ### Where
        ### S = (Delta'Sigma_Y^{-1}Delta + Sigma_Alpha^{-1})^{-1}
        ### b = (Delta'Sigma_Y^{-1}(Y - X\beta) + 0)
        covm_update = st.Delta.T.dot(st.Delta) / st.Sigma2
        covm_update += st.PsiLambdai / st.Tau2
        covm_update = la.inv(covm_update)

        resids = st.Y - st.XBetas
        mean_update = st.Delta.T.dot(resids) / st.Sigma2
        mean_update = np.dot(covm_update, mean_update)
        st.Alphas = chol_mvn(mean_update, covm_update)
        st.DeltaAlphas = np.dot(st.Delta, st.Alphas)

        ### Sample the Random Effect aspatial variance parameter
        ### P(Tau2 | .) \propto L(Y|.) \dot P(\Alpha | \lambda, Tau2)
        ###                            \dot P(Tau2) \dot P(\lambda)
        ### is
        ### IG(J/2 + a0, u'(\Psi(\lambda))^{-1}u * .5 + b0)
        bn = st.Alphas.T.dot(st.PsiLambdai).dot(st.Alphas) * .5 + st.Tau2_b0
        st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=bn)

        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = eta.T.dot(eta) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)

        ### P(Psi(\rho) | . ) \propto L(Y | .) \dot P(\rho)
        ### is
        ### |Psi(rho)|^{-1/2} exp(1/2(eta'Psi(rho)^{-1}eta * Sigma2^{-1})) * 1/(emax-emin)
        st.Lambda = self.configs.Lambda(st)
        st.PsiLambdai = st.Psi_2i(st.Lambda, st.M)


class Upper_SMA(Base_Upper_SMA):
    """
    This is a class that provides the generic structures required for the two-level variance components model with spatial moving average errors in the region level and independent errors in the lower level.

    Formally, this is the following distributional model for Y:

    Y ~ N(X * Beta, I * Sigma2 + Delta Phi_2(Lambda, Tau2) Delta')

    Where Delta is the dummy variable matrix, Rho, Sigma2 are response-level autoregressive and scale components of a spatial autoregressive covariance function, Phi_1. Lambda and Tau2 are regional-level components for spatial autoregressive covariance function, Phi_2. In this case, Phi_1 and Phi_2 are Simultaneously-autoregressive errors over weights matrices W,M:

    Phi_1(Rho, Sigma2) = I * Sigma2
    Phi_2(Lambda, Tau2) = [(I + Lambda M)(I + Lambda M)'] * Tau2

    Arguments
    ----------
    Y       :   numpy.ndarray
                The (n,1) array containing the response to be modeled
    X       :   numpy.ndarray
                The (n,p) array containing covariates used to predict the  response, Y.
    W       :   pysal.weights.W
                a spatial weights object for the n observations
    M       :   pysal.weights.W
                a spatial weights object for the J regions.
    Z       :   numpy.ndarray
                The (J,p') array of region-level covariates used to predict the response, Y.
    Delta   :   numpy.ndarray
                The (n,J) dummy variable matrix, relating observation i to region j. Required if membership is not passed.
    membership: numpy.ndarray
                The (n,) vector of labels assigning each observation to a region. Required if Delta is not passed.
    transform  : string
                 weights transform to use in modeling.
    verbose    : bool
                 whether to provide verbose output about sampling progress
    n_samples : int
                the number of samples to draw. If zero, the model will only be initialized, and later sampling can be done using model.sample
    n_jobs  :   int
                the number of parallel chains to run. Defaults to 1.
    extra_traced_param  :   list of strings
                            extra parameters in the trace to record.
    center  :   bool
                Whether to center the input data so that its mean is zero. Occasionally improves the performance of the sampler.
    scale   :   bool
                Whether to rescale the input data so that its variance is one. May dramatically improve the performance of the sampler, at the cost of interpretability of the coefficients.
    priors  :   dictionary
                A dictionary used to configure the priors for the model. This may include the following keys:
                    Betas_cov0  : prior covariance for Beta parameter vector
                                (default: I*100)
                    Betas_mean0 : prior mean for Beta parameters
                                (default: 0)
                    Sigma2_a0   : prior shape parameter for inverse gamma prior on response-level variance
                                (default: .001)
                    Sigma2_b0   : prior scale parameter for inverse gamma prior on response-level variance
                                (default: .001)
                    Tau2_a0     : prior shape parameter for inverse gamma prior on regional-level variance
                                (default: .001)
                    Tau2_b0     : prior scale parameter for inverse gamma prior on regional-level variance
                                (default: .001)
                    Log_Lambda0 : prior on Lambda, the region-level autoregressive parameter. Must be a callable function that takes a single argument and returns a single value providing the log prior likelihood.
                                (default: priors.constant)
                    Log_Rho0    : prior on Rho, the response-level autoregressive paraameter. Must be a callable function that takes a single argument and returns a single value providing the log prior likelihood.
                                (default: priors.constant)
    starting_value :    dictionary
                        A dictionary containing the starting values of the sampler. If n_jobs > 1, these starting values are perturbed randomly to induce overdispersion.

                        This dicutionary may include the following keys:
                        Betas   : starting vector of Beta parameters.
                                (default: np.zeros(p,1))
                        Alphas  : starting vector of Alpha variance components.
                                (default: np.zeros(J,1))
                        Sigma2  : starting value of response-level variance
                                (default: 4)
                        Tau2    : starting value of region-level varaince
                                (default: 4)
                        Rho     : starting value of response-level spatial autoregressive parameter
                                (default: -1/n)
                        Lambda  : starting value of region-level spatial autoregressive parameter
                                (default -1/J)
    config  :   dictionary
                A dictionary containing the configuration values for the non-Gibbs samplers for the spatial parameters. May be specified for each parameter individually (if both are in the model), or may be specified implicitly for the relevant parameter to the model. Keys may include:
                Rho_method      : string specifying whether "met" or "slice" sampling should be used for the response-level effects
                Rho_configs     : configuration options for the sampler for the response-level effects that will be used in the step method.
                Lambda_method   : string specifying whether 'met' or 'slice' sampling should be used for the region-level effects
                Lambda_configs  : configuration options for the sampler for the region-level effects that will be used in the step method.

                For options that can be in Lambda/Rho_configs, see:
                spvcm.steps.Slice, spvcm.steps.Metropolis
    truncation  :   dictionary
                    A dictionary containing the configuration values for the maximum and minimum allowable Lambda and Rho parameters. If these are not provided, the support is row-standardized by default, and the minimal eigenvalue computed for the lower bound on the parameters. *only* the single minimum eigenvalue is computed, so this is still rather efficient for large matrices. Keys may include:
                    Rho_min     : minimum value allowed for response-level autoregressive coefficient
                    Rho_max     : maximum value allowed for response-level autoregressive coefficient
                    Lambda_min  : minimum value allowed for region-level autoregressive coefficient
                    Lambda_max  : maximum value allowed for region-level autoregressive coefficient.
    """
    def __init__(self, Y, X, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', verbose=False,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 truncation=None,
                 center=False,
                 scale=False):
        _, M = verify.weights(None, M, transform=transform)
        self.M = M
        Mmat = M.sparse

        N,_ = X.shape
        if Delta is not None:
            J = Delta.shape[1]
        elif membership is not None:
            J = len(np.unique(membership))

        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        if center:
            X = verify.center(X)
        if scale:
            X = verify.scale(X)

        X = verify.covariates(X)

        self._verbose = verbose

        super(Upper_SMA, self).__init__(Y, X, Mmat, Delta,
                                       n_samples=n_samples,
                                       n_jobs = n_jobs,
                                       extra_traced_params=extra_traced_params,
                                       priors=priors,
                                       configs=configs,
                                       starting_values=starting_values,
                                       truncation=truncation)
