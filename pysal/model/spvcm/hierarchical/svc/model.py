from __future__ import division
import numpy as np
import copy
import scipy.linalg as scla
import scipy.sparse as spar
from scipy import stats
from scipy.spatial import distance as d
from .utils import explode, nexp
from .sample import sample_phi, logp_phi
from ...abstracts import Sampler_Mixin, Trace, Hashmap
from ...verify import center as verify_center, covariates as verify_covariates
from ...utils import chol_mvn
from ...steps import Metropolis, Slice

class SVC(Sampler_Mixin):
    """
    A class to initialize a spatially-varying coefficient model

    Parameters to be estimated:
    Beta    : effects at each point, distributed around Mu _\beta
    Mus     : hierarchical mean of Beta effects
    T       : global covariance matrix for effects
    Phi     : spatial dependence parameter for effects
    Tau2    : prediction error for the model

    The distance matrix will be divided by the maximum distance,
    so that it will vary between 0 and 1.

    Hyperparameters as given:
    a0          : tau location hyperparameter
    b0          : tau scale hyperparameter
    v0          : T matrix degrees of freedom parameter
    Omega       : T matrix covariance prior
    mu0         : Mean beta hyperparameter
    mu_cov0      : Beta hyperparameter for Covariance
    phi_shape0  : Phi spatial dependence shape hyperparameter
    phi_rate0   : Phi spatial dependence rate hyperparameter
    phi_scale0  : Phi spatial dependence scale hyperparameter, overrides the
                  rate parameter. always 1/phi_rate0
    """
    def __init__(self,
                 #data parameters
                 Y, X, coordinates, n_samples=1000, n_jobs=1,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 extra_traced_params = None,
                 dmetric='euclidean',
                 correlation_function=nexp,
                 verbose=False,
                 center=True,
                 rescale_dists=True):
        if center:
            X = verify_center(X)
        X = verify_covariates(X)

        N,p = X.shape
        Xs = X

        X = explode(X)
        self.state = Hashmap(X=X, Y=Y, coordinates=coordinates)
        self.traced_params = ['Betas', 'Mus', 'T', 'Phi', 'Tau2']
        if extra_traced_params is not None:
            self.traced_params.extend(extra_traced_params)
        self.trace = Trace(**{param:[] for param in self.traced_params})
        st = self.state
        self.state.correlation_function = correlation_function
        self.verbose = verbose


        st.Y = Y
        st.X = X
        st.Xs = Xs
        st.N = N
        st.p = p

        st._dmetric = dmetric
        if isinstance(st._dmetric, str):
            st.pwds = d.squareform(d.pdist(st.coordinates, metric=st._dmetric))
        elif callable(st._dmetric):
            st.pwds = st._dmetric(st.coordinates)

        st.max_dist = st.pwds.max()
        if rescale_dists:
            st.pwds = st.pwds/st.max_dist
            st._old_max = st.max_dist
            st.max_dist = 1.

        if configs is None:
            configs = dict()
        if priors is None:
            priors = dict()
        if starting_values is None:
            starting_values = dict()

        self._setup_priors(**priors)
        self._setup_starting_values(**starting_values)
        self._setup_configs(**configs)

        self._verbose = verbose
        self.cycles = 0

        if n_samples > 0:
            try:
                self.sample(n_samples, n_jobs=n_jobs)
            except (np.linalg.LinAlgError, ValueError) as e:
                Warn('Encountered the following LinAlgError. '
                     'Model will return for debugging. \n {}'.format(e))

    def _setup_priors(self, Tau2_a0=.001, Tau2_b0=.001,
                            T_v0 = 3, T_Omega0=None,
                            Mus_mean0 = None, Mus_cov0=None,
                            Phi_shape0=1, Phi_rate0=None):
        st = self.state
        st.Tau2_a0 = Tau2_a0
        st.Tau2_b0 = Tau2_b0
        st.T_v0 = T_v0
        if T_Omega0 is None:
            st.Ip = np.identity(st.p)
            T_Omega0 = .1 * st.Ip
        st.T_Omega0 = T_Omega0
        if type(Mus_mean0) in (float, int):
            Mus_mean0 = np.ones((st.p,1)) * Mus_mean0
        elif Mus_mean0 is None:
            Mus_mean0 = np.zeros((st.p, 1))
        st.Mus_mean0 = Mus_mean0
        if Mus_cov0 is None:
            Mus_cov0 = 1000*st.Ip
        st.Mus_cov0 = Mus_cov0
        st.Phi_shape0 = Phi_shape0
        if Phi_rate0 is None:
           Phi_rate0 = st.Phi_shape0 / ((-.5*st.pwds.max() / np.log(.05)))
        st.Phi_rate0 = Phi_rate0

    def _setup_starting_values(self, Phi=None, T=None,
                                     Mus=None, Betas=None, Tau2=2):
        """
        Setup initial values of the sampler for parameters

        Defaults
        ---------
        Phi     : 3*shape/rate for the parameter by default.
        T       : an identity matrix
        Mu      : zeros
        Beta    : zeros
        """
        if T is None:
            T = np.eye(self.state.p)
        self.state.T = T
        if Mus is None:
            Mus = np.zeros((1,self.state.p))
        self.state.Mus = Mus
        if Betas is None:
            Betas = np.zeros((self.state.p * self.state.N, 1))
        self.state.Betas = Betas
        if Phi is None:
            Phi = 3*self.state.Phi_shape0 / self.state.Phi_rate0
        self.state.Phi = Phi
        self.state.Tau2=Tau2

    def _setup_configs(self, Phi_method = 'met', Phi_configs=None, **uncaught):
        """
        Setup Configs of the MCMC sampled parameters in the model.
        """
        if Phi_method.lower().startswith('met'):
            method = Metropolis
        elif Phi_method.lower().startswith('slice'):
            method = Slice
        else:
            raise Exception('`Phi_method` option not understood. `{}` provided'             .format(Phi_method))

        if uncaught != dict() and Phi_configs is None:
            Phi_configs = uncaught
        elif Phi_configs is not None and uncaught == dict():
            if isinstance(Phi_configs, dict):
                Phi_configs = Phi_configs
            else:
                raise TypeError('Type of `Phi_configs` not understood. Must be'
                                'dict containing the configuration for the MCMC step sampling Phi. Recieved object of type:'
                                '\n{}'.format(type(Phi_configs)))
        elif Phi_configs is None and uncaught == dict():
            Phi_configs = uncaught
        else:
            raise Exception('Uncaught options {} passed in addition to '
                            '`Phi_configs` {}.'.format(uncaught, Phi_configs))

        self.configs = Hashmap()
        self.configs.Phi = method('Phi', logp_phi, **Phi_configs)


    def _finalize(self):
        """
        Compute derived quantities that make the sampling loop more efficient.
        Once computed, the priors are considered "set."
        """
        st = self.state
        st = self.state
        st.In = np.identity(st.N)
        st.XtX = st.X.T.dot(st.X)
        st.iota_n = np.ones((st.N,1))
        st.Xty = st.X.T.dot(st.Y)
        st.np2n = np.zeros((st.N * st.p, st.N))
        for i in range(st.N):
            st.np2n[i*st.p:(i+1)*st.p, i] = 1
        st.np2p = np.vstack([np.eye(st.p) for _ in range(st.N)])
        st.Mus_cov0i = scla.inv(st.Mus_cov0)
        st.Mus_kernel_prior = np.dot(st.Mus_cov0i, st.Mus_mean0)
        st.Tau2_an = st.Tau2_a0 + st.N/2.
        st.T_vn = st.T_v0 + st.N

    def _fuzz_starting_values(self):
        super(SVC, self)._fuzz_starting_values()
        self.state.Phi += np.random.uniform(0,10)
        self.state.Mus += np.random.normal(0,10,size=self.state.Mus.shape)
        self.state.Tau2 += np.random.uniform(0,10)

    def _iteration(self):
        """
        Conduct one iteration of a Gibbs sampler for the self using the state
        provided.
        """
        st = self.state

        ## Tau, EQ 3 in appendix of Wheeler & Calder
        ## Inverse Gamma w/ update to scale, no change to dof
        y_Xbeta = st.Y - st.X.dot(st.Betas)
        st.Tau2_bn = st.Tau2_b0 + .5 * y_Xbeta.T.dot(y_Xbeta)
        st.Tau2 = stats.invgamma.rvs(st.Tau2_an, scale=st.Tau2_bn)

        ##covariance: T, EQ 4 in appendix of Wheeler & Calder
        ## inverse wishart w/ update to covariance matrix, no change to dof
        st.H = st.correlation_function(st.Phi, st.pwds)
        st.Hinv = scla.inv(st.H)
        st.tiled_Hinv = np.linalg.multi_dot([st.np2n, st.Hinv, st.np2n.T])
        st.tiled_Mus = np.kron(st.iota_n, st.Mus.reshape(-1,1))
        st.info = (st.Betas - st.tiled_Mus).dot((st.Betas - st.tiled_Mus).T)
        st.kernel = np.multiply(st.tiled_Hinv, st.info)
        st.covm_update = np.linalg.multi_dot([st.np2p.T, st.kernel, st.np2p])
        st.T = stats.invwishart.rvs(df=st.T_vn, scale=(st.covm_update + st.T_Omega0))

        ##mean hierarchical effects: mu_\beta, in EQ 5 of Wheeler & Calder
        ##normal with both a scale and a location update, priors don't change
        #compute scale of mu_\betas
        st.Sigma_beta = np.kron(st.H, st.T)
        st.Psi = np.linalg.multi_dot((st.X, st.Sigma_beta, st.X.T)) + st.Tau2 * st.In
        Psi_inv = scla.inv(st.Psi)
        S_notinv_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Xs))
        S = scla.inv(st.Mus_cov0i + S_notinv_update)

        #compute location of mu_\betas
        mkernel_update = np.linalg.multi_dot((st.Xs.T, Psi_inv, st.Y))
        st.Mus_mean = np.dot(S, (mkernel_update + st.Mus_kernel_prior))

        #draw them using cholesky decomposition: N(m, Sigma) = m + chol(Sigma).N(0,1)
        st.Mus = chol_mvn(st.Mus_mean, S)
        st.tiled_Mus = np.kron(st.iota_n, st.Mus)

        ##effects \beta, in equation 6 of Wheeler & Calder
        ##Normal with an update to both scale and location, priors don't change

        #compute scale of betas
        st.Tinv = scla.inv(st.T)
        st.kronHiTi = np.kron(st.Hinv, st.Tinv)
        Ai = st.XtX / st.Tau2 + st.kronHiTi
        A = scla.inv(Ai)

        #compute means of betas
        C = st.Xty / st.Tau2 + np.dot(st.kronHiTi, st.tiled_Mus)
        st.Betas_means = np.dot(A, C)
        st.Betas_cov = A

        #draw them using cholesky decomposition
        st.Betas = chol_mvn(st.Betas_means, st.Betas_cov)

        # local nonstationarity parameter Phi, in equation 7 in Wheeler & Calder
        # sample using metropolis
        st.Phi = self.configs.Phi(st)
