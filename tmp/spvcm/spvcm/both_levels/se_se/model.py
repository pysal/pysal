from ..generic import Base_Generic
from ..generic.sample import logp_rho_prec, logp_lambda_prec
from ...utils import se_covariance, se_precision
from ... import verify
import numpy as np

class Base_SESE(Base_Generic):
    """
    All arguments to this function ar documented in SESE
    """
    def __init__(self, Y, X, W, M, Delta,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 truncation=None):
        super(Base_SESE, self).__init__(Y, X, W, M, Delta,
                                        n_samples=0, n_jobs=n_jobs,
                                        extra_traced_params=extra_traced_params,
                                        priors=priors,
                                        configs=configs,
                                        starting_values=starting_values,
                                        truncation=truncation)
        self.state.Psi_1 = se_covariance
        self.state.Psi_2 = se_covariance
        self.state.Psi_1i = se_precision
        self.state.Psi_2i = se_precision

        self.configs.Rho.logp = logp_rho_prec
        self.configs.Lambda.logp = logp_lambda_prec

        try:
            self.sample(n_samples, n_jobs =n_jobs)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))

class SESE(Base_SESE):
    """
    This is a class that provides the generic structures required for the two-level variance components model with simultaneous autregressive errors in both levels.

    Formally, this is the following distributional model for Y:

    Y ~ N(X * Beta, Phi_1(Rho, Sigma2) + Delta Phi_2(Lambda, Tau2) Delta')

    Where Delta is the dummy variable matrix, Rho, Sigma2 are response-level autoregressive and scale components of a spatial autoregressive covariance function, Phi_1. Lambda and Tau2 are regional-level components for spatial autoregressive covariance function, Phi_2. In this case, Phi_1 and Phi_2 are Simultaneously-autoregressive errors over weights matrices W,M:

    Phi_1(Rho, Sigma2) = [(I - Rho W)'(I - Rho W)]^{-1} * Sigma2
    Phi_2(Lambda, Tau2) = [(I - Lambda M)'(I - Lambda M)]^{-1} * Tau2

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
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
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
        W,M = verify.weights(W, M, transform=transform)
        self.M = M
        N,_ = X.shape
        J = M.n
        Mmat = M.sparse
        Wmat = W.sparse

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

        super(SESE, self).__init__(Y, X, Wmat, Mmat, Delta,
                                   n_samples=n_samples,
                                   n_jobs = n_jobs,
                                   extra_traced_params=extra_traced_params,
                                   priors=priors,
                                   configs=configs,
                                   starting_values=starting_values,
                                   truncation=truncation)
