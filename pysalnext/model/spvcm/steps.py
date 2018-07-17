from __future__ import division
import numpy as np
import scipy.stats as stats
import copy
MAX_SLICE = 1000

#################################
# STANDALONE SAMPLING FUNCTIONS #
#################################

def inversion(pdvec, grid):
    """
    sample from a probability distribution vector, according to a grid of values

    Parameters
    -----------
    pdvec   :   np.ndarray
                a vector of point masses that must sum to one. in theory, an
                approximation of a continuous pdf
    grid    :   np.ndarray
                a vector of values over which pdvec is evaluated. This is the
                bank of discrete values against which the new value is drawn
    """
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    a = 0
    while True:
        a += 1
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

def metropolis(state, current, proposal, logp, jump):
    """
    Sample using metropolis hastings in its simplest form

    Parameters
    ----------
    state   :   Hashmap
                state required to evaluate the logp of the parameter
    current :   float/int
                current value of the parameter
    proposal:   scipy random distribution
                distribution that has both `logpdf` and `rvs` methods
    logp    :   callable(state, value)
                function that can compute the current log of the probability
                distribution of the value provided conditional on the state
    jump    :   int
                the scaling factor for the proposal distribution
    Returns
    --------
    new (or current) parameter value, and boolean indicating whether or not a
    new proposal was accepted.
    """
    #try:
    current_logp = logp(state, current)
    new_val = proposal.rvs(loc=current, scale=jump)
    new_logp = logp(state, new_val)
    forwards = proposal.logpdf(new_val, loc=current, scale=jump)
    backward = proposal.logpdf(current, loc=new_val, scale=jump)

    hastings_factor = backward - forwards
    r = new_logp - current_logp + hastings_factor

    r = np.min((0, r))
    u = np.log(np.random.random())
    #except:
    #    print(current)
    #    print(new_val)
    #    print(new_logp)
    #    print(current_logp)
    #    print(backward)
    #    print(forwards)
    #    print(r)
    #    print(u)
    #    raise

    if u < r:
        outval = new_val
        accepted = True
    else:
        outval = current
        accepted = False
    return outval, accepted

def slicer(state, current, logp, width, adapt=10):
    """
    Implements slice sampling on a bounded log-concave parameter, as according to Neal 2003.

    Arguments
    ---------
    state   :   Hashmap/dict or tuple/list/iterable
                a collection of attributes/arguments required by the ``
                that are not `current`, the current value of the parameter
                being sampled. Is passed as a single argument to `logp`.
    current :   Float/Int
                current value of the parameter being sampled.
                NOTE: This sampler is a univariate slice sampler.
    logp    :   Callable
                function that takes (`state`, `current`) and provides the log
                probability density function to be sampled.
    width   :   float/int
                width of the slice to use when sampling.
    adapt   :   int
                weight to place on initial slice widths. Computes the moving average of the past `adapt` widths and the current width.

    """
    current_logp = logp(state, current)
    n_iterations = 0
    # p. 712 of Neal defines this auxiliary variable on the log scale
    slice_height = current_logp - np.random.exponential()

    left = current - np.random.uniform(0,width)
    right = left + width

    while slice_height < logp(state, left):
        left -= .1*width
    while slice_height < logp(state, right):
        right += .1*width


    while True:
        candidate = np.random.uniform(left, right)
        cand_logp = logp(state, candidate)
        if slice_height <= cand_logp:
            break
        if candidate > current:
            right = candidate
        elif candidate < current:
            left = candidate
        candidate = np.random.uniform(left, right)
        n_iterations += 1
        if n_iterations > MAX_SLICE:
            warn('Slicing is failing to find an effective candidate. '
                 'Using a metropolis update.', stacklevel=2)
            return metropolis(state, current, logp, configs)

    if adapt > 0:
        MA = [np.abs(left - right)]
        MA.extend([width for _ in range(adapt)])
        width = np.mean(MA)

    return candidate, True, width

def hmc(state, current_value, current_momentum, logp, dlogp):
    raise NotImplementedError

def ars(state, current_value):
    raise NotImplementedError

###############################
# SAMPLING CLASS DECLARATIONS #
###############################

class AbstractStep(object):
    """
    Standin for an abstract step
    """
    def __init__(self, varname):
        super(AbstractStep, self).__init__()
        self.varname = varname

    @property
    def _idempotent(self):
        return False

    def __call__(self, state):
        raise NotImplementedError

    def __draw__(self, state):
        return self(state)

class Gibbs(AbstractStep):
    """
    Sample directly from a given conditional log posterior distribution using the given kernel. Currently unused.
    """
    def __init__(self, varname, kernel):
        super(Gibbs, self).__init__(varname)
        self.kernel = kernel

    @property
    def _idempotent(self):
        return False

    def __call__(self, state):
        return self.kernel(state)

class Metropolis(AbstractStep):
    """
    Sample the given Logp using metroplis sampling

    Arguments
    ---------
    varname :   string
                name of variable to compute.
    logp    :   function
                function that, takes two arguments, value and state, and returns the log of the probability distribution being sampled.
    jump    :   float
                scale parameter to use in the metropolis proposal distribution
    proposal:   scipy random distribution
                distribution that has both `logpdf` and `rvs` methods
    adapt_step: float > 1
                the rate at which to adjust the jump when using tuned metropolis.
    ar_low  :   float in [0,1]
                lower bound on the acceptance rate of the metropolis sampler
    ar_hi   :   float in [0,1]
                upper bound on acceptance rate of the metrpolis sampler.
    max_tuning: int
                maximum number of iterations to tune the sampler.
    debug   :   bool
                flag denoting whether to store parameters in each iteration in the _cache.
    """
    def __init__(self, varname, logp, jump = 1, proposal = stats.norm,
                       adapt_step=1.01, ar_low = .2, ar_hi = .25, max_tuning = 0, debug=False):
        super(Metropolis, self).__init__(varname)
        self.jump = jump
        self.adapt_step = adapt_step if adapt_step >= 1 else 1/adapt_step
        self.ar_low = ar_low
        self.ar_hi = ar_hi
        self.max_tuning = max_tuning
        self.proposal = proposal
        self.accepted = 0
        self.cycles = 0
        self.logp = logp
        self.debug = debug
        if debug:
            self._cache = []

    @property
    def _idempotent(self):
        return True

    def __call__(self, state):
        orig_val = copy.deepcopy(getattr(state, self.varname))
        new, accepted = metropolis(state, orig_val,
                                    self.proposal, self.logp, self.jump)
        self.accepted += int(accepted)
        self.cycles += 1
        self.is_tuning = self.cycles <= self.max_tuning
        if self.is_tuning:
            acceptance_rate = self.accepted / self.cycles
            if acceptance_rate < self.ar_low:
                self.jump /= self.adapt_step
            if acceptance_rate > self.ar_hi:
                self.jump *= self.adapt_step
        if self.debug:
            self._cache.append(dict(jump = self.jump,
                                    current_logp = self.logp(state, orig_val),
                                    new_logp = self.logp(state, new),
                                    accepted = accepted))
        return new

class Slice(AbstractStep):
    """
    Sample the given Logp using slice sampling, of Neal (2003).

    Arguments
    ------------
    varname :   string
                name of variable to compute.
    logp    :   function
                function that, takes two arguments, value and state, and returns the log of the probability distribution being sampled.
    width   :   float
                width of the level set to use.
    adapt   :   int
                number of iterations previous to use in averaging the slice width.
    debug   :   bool
                whether or not to save the parameters in each step to the _cache attribute.
    """
    def __init__(self, varname, logp, width=1, adapt=0, debug=False):
        super(Slice, self).__init__(varname)
        self.logp = logp
        self.width = width
        self.adapt = adapt
        self.debug = debug
        if debug:
            self._cache = []

    @property
    def _idempotent(self):
        return True

    def __call__(self, state):
        new, _, width = slicer(state, getattr(state, self.varname),
                               self.logp, self.width, self.adapt)
        self.width = width
        if self.debug:
            self._cache.append(dict(width=self.width,
                                    logp = self.logp(state, new)))
        return new
