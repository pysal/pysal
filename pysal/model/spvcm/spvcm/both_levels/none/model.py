from __future__ import division

import numpy as np
import scipy.stats as stats
import copy

from numpy import linalg as la
from warnings import warn as Warn
from ...abstracts import Sampler_Mixin, Hashmap, Trace
from ... import verify
from ...utils import ind_covariance, chol_mvn


SAMPLERS = ['Alphas', 'Betas', 'Sigma2', 'Tau2']

class Base_MVCM(Sampler_Mixin):
    """
    The class that actually ends up setting up the MVCM model. Sets configs,
    data, truncation, and starting parameters, and then attempts to apply the
    sample function n_samples times to the state.
    """
    def __init__(self, Y, X, Delta,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None):
        super(Base_MVCM, self).__init__()

        N, p = X.shape
        _N, J = Delta.shape
        self.state = Hashmap(**{'X':X, 'Y':Y, 'Delta':Delta,
                           'N':N, 'J':J, 'p':p })
        self.traced_params = copy.deepcopy(SAMPLERS)
        extras = extra_traced_params
        if extras is not None:
            self.traced_params.extend(extra_tracked_params)
        hashmaps = [Hashmap(**{k:[] for k in self.traced_params})]*n_jobs
        self.trace = Trace(*hashmaps)

        if priors is None:
            priors = dict()
        if starting_values is None:
            starting_values = dict()

        self._setup_priors(**priors)
        self._setup_starting_values(**starting_values)

        self.cycles = 0
        self.configs = None


        try:
            self.sample(n_samples, n_jobs=n_jobs)
        except (np.linalg.LinAlgError, ValueError) as e:
            Warn('Encountered the following LinAlgError. '
                 'Model will return for debugging. \n {}'.format(e))


    def _setup_priors(self, Sigma2_a0=.001, Sigma2_b0 = .001,
                      Betas_cov0 = None, Betas_mean0 = None,
                      Tau2_a0 = .001, Tau2_b0 = .0001):
        ## Prior specifications
        st = self.state
        st.Sigma2_a0 =  Sigma2_a0
        st.Sigma2_b0 = Sigma2_b0
        if Betas_cov0 is None:
            Betas_cov0 = np.eye(self.state.p) * 100
        st.Betas_cov0 = Betas_cov0
        if Betas_mean0 is None:
            Betas_mean0 = np.zeros((self.state.p, 1))
        st.Betas_mean0 = Betas_mean0
        st.Tau2_a0 = Tau2_a0
        st.Tau2_b0 = Tau2_b0

    def _finalize(self):
        """
        This computes derived properties of hyperparameters that do not change
        over iterations. This is called one time before sampling.
        """
        st = self.state

        st.XtX = np.dot(st.X.T, st.X)
        st.DeltatDelta = np.dot(st.Delta.T, st.Delta)

        st.Betas_cov0i = np.linalg.inv(st.Betas_cov0)
        st.Betas_covm = np.dot(st.Betas_cov0, st.Betas_mean0)
        st.Sigma2_an = st.N / 2 + st.Sigma2_a0
        st.Tau2_an = st.J / 2 + st.Tau2_a0
        st.In = np.identity(st.N)
        st.Ij = np.identity(st.J)
        st.DeltaAlphas = np.dot(st.Delta, st.Alphas)
        st.XBetas = np.dot(st.X, st.Betas)

        st.Psi_1 = ind_covariance
        st.Psi_2 = ind_covariance

        st.PsiRho = st.In
        st.PsiLambda = st.Ij

        st.PsiSigma2 = st.In * st.Sigma2
        st.PsiSigma2i = la.inv(st.PsiSigma2)
        st.PsiTau2 = st.Ij * st.Tau2
        st.PsiTau2i = la.inv(st.PsiTau2)

        st.PsiRhoi = la.inv(st.PsiRho)
        st.PsiLambdai = la.inv(st.PsiLambda)

    def _setup_starting_values(self, Betas=None, Alphas=None,
                                Sigma2=4, Tau2=4):
        """
        Set abrbitrary starting values for the Metropolis sampler
        """
        if Betas is None:
            Betas = np.zeros((self.state.p, 1))
        if Alphas is None:
            Alphas = np.zeros((self.state.J, 1))
        self.state.Betas = Betas
        self.state.Alphas = Alphas
        self.state.Sigma2 = Sigma2
        self.state.Tau2 = Tau2

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
        covm_update += st.PsiTau2i
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
        st.PsiTau2 = st.Ij * st.Tau2
        st.PsiTau2i = la.inv(st.PsiTau2)

        ### Sample the response aspatial variance parameter
        ### P(Sigma2 | . ) \propto L(Y | .) \dot P(Sigma2)
        ### is
        ### IG(N/2 + a0, eta'Psi(\rho)^{-1}eta * .5 + b0)
        ### Where eta is the linear predictor, Y - X\beta + \DeltaAlphas
        eta = st.Y - st.XBetas - st.DeltaAlphas
        bn = eta.T.dot(st.PsiRhoi).dot(eta) * .5 + st.Sigma2_b0
        st.Sigma2 = stats.invgamma.rvs(st.Sigma2_an, scale=bn)

class MVCM(Base_MVCM):
    """
    This is a class that provides the generic structures required for the two-level variance components model

    Formally, this is the following distributional model for Y:

    Y ~ N(X * Beta, I_n * Sigma2 + Delta Delta' Tau2)

    Where Delta is the dummy variable matrix, Sigma2 is the response-level variance, and Tau2 is the region-level variance. 

    Arguments
    ----------
    Y       :   numpy.ndarray
                The (n,1) array containing the response to be modeled
    X       :   numpy.ndarray
                The (n,p) array containing covariates used to predict the  response, Y.
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

                For options that can be in Lambda/Rho_configs, see:
                spvcm.steps.Slice, spvcm.steps.Metropolis
    """
    def __init__(self, Y, X, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', n_samples=1000, n_jobs=1,
                 verbose=False,
                 extra_traced_params = None,
                 priors=None,
                 starting_values=None,
                 center=False,
                 scale=False
                 ):

        N, _ = X.shape
        if Delta is not None:
            _,J = Delta.shape
        elif membership is not None:
            J = len(np.unique(membership))
        else:
            raise UserWarning("No Delta matrix nor membership classification provided. Refusing to arbitrarily assign units to upper-level regions.")
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
        super(MVCM, self).__init__(Y, X, Delta, n_samples=n_samples,
                                   n_jobs=n_jobs,
                                   extra_traced_params = extra_traced_params,
                                   priors=priors,
                                   starting_values=starting_values)
