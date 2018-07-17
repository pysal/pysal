from __future__ import division
import numpy as _np
import pandas as _pd
from .utils import thru_op as _thru_op
from .abstracts import Trace as _Trace, Hashmap as _Hashmap, _maybe_hashmap, _copy_hashmaps
from warnings import warn as _Warn
import copy as _copy

__all__ = ['summarize', 'mcse', 'psrf', 'geweke', 'hpd_interval', 'effective_size']

try:
    from rpy2.rinterface import RRuntimeError
    from rpy2.robjects.packages import importr
    from rpy2.robjects.numpy2ri import numpy2ri
    import rpy2.robjects as ro
    ro.conversion.py2ri = numpy2ri
    _coda = importr('coda')
    _HAS_CODA = True
    _HAS_RPY2 = True
except (ImportError, LookupError):
    _HAS_CODA = False
    _HAS_RPY2 = False
except RRuntimeError:
    _HAS_CODA = False
    _HAS_RPY2 = True

#############
# Summarize #
#############

def summarize(model = None, trace = None, chain=None, varnames=None,
              level=0):
    """
    Summarize a trace object, providing its mean, median, HPD,
    standard deviation, and effective size.

    Arguments
    ---------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples
                of p covariates.
    varnames:   str or list of str
                set of variates to extract from the model or trace to to compute the
                statistic.
    level   :   int
                ordered in terms of how much information reduction occurs. a level 0 summary
                provides the output for each chain. A level 1 summary provides output
                grouped over all chains.
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    dfs = trace.to_df()
    if isinstance(dfs, list):
        multi_index = ['Chain_{}'.format(i) for i in range(len(dfs))]
        df = _pd.concat(dfs, axis=1, keys=multi_index)
    else:
        df = _pd.concat((dfs,), axis=1, keys=['Chain_0'])
    df = df.describe().T[['count', 'mean', '50%', 'std']]
    HPDs = hpd_interval(trace=trace, p=.95)
    if _HAS_CODA:
        ESS = effective_size(trace=trace, use_R=True)
    else:
        _Warn('Computing effective sample size may take a while due to statsmodels.tsa.AR.',
             stacklevel=2)
        ESS = effective_size(trace=trace, use_R=False)
    flattened_HPDs = []
    flattened_ESSs = []
    if isinstance(HPDs, dict):
        HPDs = [HPDs]
    if isinstance(ESS, dict):
        ESS = [ESS]
    for i_chain, chain in enumerate(HPDs):
        this_HPD = dict()
        this_ESS = dict()
        for key, val in chain.items():
            if isinstance(val, list):
                for i, hpd_tuple in enumerate(val):
                    name = '{}_{}'.format(key, i)
                    this_HPD.update({name: hpd_tuple})
                    this_ESS.update({name: ESS[i_chain][key][i]})
            else:
                this_HPD.update({key: val})
                this_ESS.update({key: ESS[i_chain][key]})
        flattened_HPDs.append(this_HPD)
        flattened_ESSs.append(this_ESS)
    df['HPD_low'] = None
    df['HPD_high'] = None
    df['N_effective'] = None
    for i, this_chain_HPD in enumerate(flattened_HPDs):
        this_chain_ESS = flattened_ESSs[i]
        outer_key = 'Chain_{}'.format(i)
        keys = [(outer_key, inner_key) for inner_key in this_chain_HPD.keys()]
        lows, highs = zip(*[this_chain_HPD[key[-1]] for key in keys])
        n_eff = [this_chain_ESS[key[-1]] for key in keys]
        df.ix[keys, 'HPD_low'] = lows
        df.ix[keys, 'HPD_high'] = highs
        df.ix[keys, 'N_effective'] = n_eff
    df['median'] = df['50%']
    df['N_iters'] = df['count'].apply(int)
    df['N_effective'] = df['N_effective'].apply(round)
    df.drop('count', axis=1, inplace=True)
    df['AR_loss'] = (df['N_iters'] - df['N_effective'])/df['N_iters']
    df = df[['mean', 'HPD_low', 'median', 'HPD_high', 'std',
             'N_iters', 'N_effective', 'AR_loss']]
    if level > 0:
        df = df.unstack()
        grand_mean = df['mean'].mean(axis=0)
        lowest_HPD = df['HPD_low'].min(axis=0)
        grand_median = df['median'].median(axis=0)
        highest_HPD = df['HPD_high'].max(axis=0)
        std = df['std'].mean(axis=0)
        neff = df['N_effective'].sum(axis=0)
        N = df['N_iters'].sum(axis=0)
        df = _pd.concat([grand_mean, lowest_HPD, grand_median,
                        highest_HPD, std, N, neff], axis=1)
        df.columns = ['grand_mean', 'min_HPD', 'grand_median', 'max_HPD', 'std',
                      'sum(N_iters)', 'sum(N_effective)']
    return df

#####################################
# Potential Scale Reduction Factors #
#####################################


def _gelman_rubin(chain):
    """
    Computes the original potential scale reduction factor from the
     1992 paper by Gelman and Rubin:

    \sqrt{\frac{\hat{V}}{W} * \frac{dof}{dof-2}}

    where \hat{V} is a corrected estimate of the chain variance, composed of within- and between-chain variance components, W,B:

    V_hat = W*(1-1/n) + B/n + B/(n*m)

    and the degrees of freedom terms are:
    dof = 2 * V_hat**2 / (Var(W) + Var(B) + 2Cov(W,B)).

    The equations of the variance and covariance are drawn directly from the original paper. This implementation should come close to the implementation in the R CODA package, which computes the same normalization factor.

    If the chain is multivariate, it computes the statistic for each element of the multivariate chain.
    """
    m,n = chain.shape[0:2]
    rest = chain.shape[2:]
    if len(rest) == 0:
        chain_vars = chain.var(axis=1, ddof=1)
        chain_means = chain.mean(axis=1)
        grand_mean = chain.mean()

        W = _np.mean(chain_vars)
        B = _np.var(chain_means, ddof=1)*n
        sigma2_hat = W*(1-(1/n)) + B/n
        V_hat = sigma2_hat + B/(n*m)
        t_scale = _np.sqrt(V_hat)

        #  not sure if the chain.var(axis=1, ddof=1).var() is right.
        var_W = (1-(1/n))**2 * chain_vars.var(ddof=1) / m
        var_B = ((m+1)/(m*n))**2 * (2*B**2) / (m-1)
        cov_s2xbar2 = _np.cov(chain_vars, chain_means**2, ddof=1)[0, 1]
        cov_s2xbarmu = 2 * grand_mean * _np.cov(chain_vars, chain_means, ddof=1)[0, 1]
        cov_WB = (m+1)*(n-1)/(m*n**2)*(m/n)*(cov_s2xbar2 - cov_s2xbarmu)

        t_dof = 2 * V_hat**2 / (var_W + var_B + 2 * cov_WB)

        psrf = _np.sqrt((V_hat / W) * t_dof/(t_dof - 2))
        return psrf
    else:
        return [_gelman_rubin(ch.T) for ch in chain.T]


def _brooks_gelman_rubin(chain):
    """
    Computes the Brooks and Gelman psrf in equation 1.1, pg. 437 of the Brooks and Gelman article.

    This form is:
    (n_chains + 1) / n_chains * (sigma2_hat / W) - (n-1)/(nm)
    where Sigma2_hat is the unbiased estimator for aggregate variance:
    (n-1)/n * W + B/n

    If the chain is multivariate, this computes the univariate version over all elements of the chain.
    """
    m,n = chain.shape[0:2]
    rest = chain.shape[2:]
    if len(rest) == 0:
        chain_vars = chain.var(axis=1, ddof=1)
        chain_means = chain.mean(axis=1)
        grand_mean = chain.mean()

        W = _np.mean(chain_vars)
        B = _np.var(chain_means, ddof=1)*n

        sigma2_hat = ((n-1)/n) * W + B/n
        Rhat = (m+1)/m * (sigma2_hat / W) - ((n-1)/(m*n))
        return _np.sqrt(Rhat)
    else:
        return [_brooks_gelman_rubin(ch.T) for ch in chain.T]

_psrf = dict([('brooks', _brooks_gelman_rubin), ('original',_gelman_rubin)])

def psrf(model = None, trace=None, chain=None, autoburnin=True,
         varnames=None, method='brooks'):
    """
    Wrapper to compute the potential scale reduction factor for
    a trace object or an arbitrary chain from a MCMC sample.

    Arguments
    ----------
    trace       :  Trace
                   A trace object that contains more-than-one chain.
    chain       :  np.ndarray
                   An array with at least two dimensions where the indices are:
                   (m,n[, k]), where m is the number of traces and n is the number of iterations. If the parameter is k-variate, then the trailing dimension must be k.
    autoburnin  :  boolean
                   a flat denoting whether to automatically slice the chain at its midpoint, computing the psrf for only the second half.
    varnames    :  string or list of strings
                   collection of the names of variables to compute the psrf.
    method      :  string
                   the psrf statistic to be used. Recognized options:
                   - 'brooks' (default): the 1998 Brooks-Gelman-Rubin psrf
                   - 'original': the 1992 Gelman-Rubin psrf
    """
    if model is not None:
        trace = model.trace
    if trace is not None and varnames is None:
        varnames = trace.varnames
    elif chain is not None and varnames is None:
        varnames = ['parameter']
    elif chain is not None and varnames is not None:
        try:
            assert len(varnames) == 1
        except AssertionError:
            raise UserWarning('Multiple chains outside of a trace '
                              'are not currently supported')
    out = dict()
    for param in varnames:
        if chain is not None:
            this_chain = chain
            m,n = chain.shape[0:2]
            rest = chain.shape[2:]
        else:
            this_chain = trace[param]
            m,n = this_chain.shape[0:2]
            rest = this_chain.shape[2:]
        this_chain = this_chain[:,-n//2:,]
        out.update({param:_psrf[method](this_chain)})
    return out

######################
# Geweke Diagnostics #
######################

def geweke(model = None, trace=None, chain=None,
           drop_frac=.1, hold_frac=.5, n_bins=50,
           varnames=None, variance_method='ar', **ar_kw):
    """
    This computes the plotting version of Geweke's diagnostic for a given trace. The iterative version is due to Brooks. This implementation mirrors that in the `R` `coda` package.

    Arguments
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples of p covariates.
    drop_frac:  float
                the number of observations to drop each step from the first (1-`keep_frac`)% of the chain
    keep_frac:  float
                the comparison group of observations used to compare the to
                the bins over the (1 - keep_frac)% of the chain
    n_bins  :   int
                number of bins to divide the first (1 - keep_frac)% of the chain
                into.
    varnames:   string or list of strings
                name or list of names of parameters to which the diagnostic should be applied.
    variance_method: str
                name of the variance method to be used. The default, `ar0`, uses the spectral density at lag 0, which is also used in CODA. This corrects for serial correlation in the variance estimate. The alternative, `naive`, is simply the sample variance.
    ar_kw   :   dict/keyword arguments
                If provided, must contain `spec_kw` and `fit_kw`. `spec_kw` is a dictionary of keyword arguments passed to the statsmodels AR class, and `fit_kw` is a dictionary of arguments passed to the subsequent AR.fit() call.
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    varnames = trace.varnames
    variance_function = _geweke_variance[variance_method]
    all_stats = []
    for i, chain in enumerate(trace.chains):
        all_stats.append(dict())
        for var in varnames:
            data = _np.squeeze(trace[i,var])
            if data.ndim > 1:
                n,p = data.shape[0:2]
                rest = data.shape[2:0]
                if len(rest) == 0:
                    data = data.T
                elif len(rest) == 1:
                    data = data.reshape(n,p*rest[0]).T
                else:
                    raise Exception('Parameter "{}" shape not understood.'                  ' Please extract, shape it, and pass '
                                    ' as its own chain. '.format(var))
            else:
                data = data.reshape(1,-1)
            stats = [_geweke_vector(datum, drop_frac, hold_frac, n_bins=n_bins,
                                    varfunc=variance_function)
                    for datum in data]
            if len(stats) > 1:
                results = {"{}_{}".format(var, i):stat for i,stat in
                            enumerate(stats)}
            else:
                results = {var:stats[0]}
            all_stats[i].update(results)
    return all_stats

def _geweke_map(model = None, trace=None, chain=None,
           drop_frac=.1, hold_frac=.5, n_bins=50,
           varnames=None, variance_method='ar', **ar_kw):
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_geweke_vector, drop=drop_frac, hold=hold_frac, n_bins=n_bins,
                      varfunc=_geweke_variance[variance_method])
    return stats

def _geweke_vector(data, drop, hold, n_bins, **kw):
    """
    Compute a vector of geweke statistics over the `data` vector. This proceeds like the `R` `CODA` package's geweke statistic. The first half of the data vector is split into `n_bins` segments. Then, the Geweke statistic is repeatedly computed over subsets of the data where a bin is dropped each step. This results in `n_bins` statistics.
    """
    in_play = (len(data)-1)//2
    to_drop = _np.linspace(0, in_play, num=n_bins).astype(int)
    return _np.squeeze([_geweke_statistic(data[drop_idx:], drop, hold, **kw)
                       for drop_idx in to_drop])

def _geweke_statistic(data, drop, hold, varfunc=None):
    """
    Compute a single geweke statistic, defining sets A, B:

    mean_A - mean_B / (var(A) + var(B))**.5

    where A is the first `drop` % of the `data` vector, B is the last `hold` % of the data vector.

    the variance function, `varfunc`, is the spectral density estimate of the variance.
    """
    if varfunc is None:
        varfunc = _spectrum0_ar
    hold_start = _np.floor((len(data)-1) * hold).astype(int)
    bin_width = _np.ceil((len(data)-1)*drop).astype(int)

    drop_data = data[:bin_width]
    hold_data = data[hold_start:]

    drop_mean = drop_data.mean()
    drop_var = varfunc(drop_data)
    n_drop = len(drop_data)

    hold_mean = hold_data.mean()
    hold_var = varfunc(hold_data)
    n_hold = len(hold_data)

    return ((drop_mean - hold_mean) / _np.sqrt((drop_var / n_drop)
                                            +(hold_var / n_hold)))

def _naive_var(data, *_, **__):
    """
    Naive variance computation of a time `x`, ignoring dependence between the
    variance within different windows
    """
    return _np.var(data, ddof=1)

def _spectrum0_ar(data, spec_kw=dict(), fit_kw=dict()):
    """
    The corrected spectral density estimate of time series variance,
    as applied in CODA. Written to replicate R, so defaults change.
    Note: this is very slow when there is a lot of data.
    """
    try:
        from statsmodels.api import tsa
    except ImportError:
        raise ImportError('Statsmodels is required to use the AR(0) '
                           ' spectral density estimate of the variance.')
    if fit_kw == dict():
        fit_kw['ic']='aic'
        N = len(data)
        # R uses the smaller of N-1 and 10*log10(N). We should replicate that.
        maxlag = N-1 if N-1 <= 10*_np.log10(N) else 10*_np.log(N)
        fit_kw['maxlag'] = int(_np.ceil(maxlag))
    ARM = tsa.AR(data, **spec_kw).fit(**fit_kw)
    alphas = ARM.params[1:]
    return ARM.sigma2 / (1 - alphas.sum())**2

_geweke_variance = dict()
_geweke_variance['ar'] = _spectrum0_ar
_geweke_variance['naive'] = _naive_var

##################
# Effective Size #
##################

def effective_size(model=None, trace=None, chain=None, varnames=None,
                    use_R=False):
    """
    Compute the effective size of a trace, accounting for serial autocorrelation.
    This statistic is:

    N * var(x)/spectral0(x)

    where spectral0(x) is the spectral density of x at lag 0, an
    autocorrelation-adjusted estimate of the sequence variance.

    NOTE: the backend argument defaults to estimating the effective_size in
    python. But, the statsmodels.tsa.AR required for the spectral density
    estimate is *slow* for large chains. If you have a properly configured R
    installation with the python package `rpy2` and the R package `coda` installed,
    you can opt to pass through to CODA by passing `use_R=True`.

    Arguments
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples
                of p covariates.
    varnames:   str or list of str
                set of variates to extract from the model or trace to to compute the
                statistic.
    use_R   :   bool (default: False)
                option to drop the computation of the effective size down to R's CODA
                package. Requires: rpy2, working R installation, CODA R package
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_effective_size, use_R=use_R)
    return stats if len(stats) > 1 else stats[0]

def _effective_size(x, use_R=False):
    """
    Compute the effective size from a given flat array

    Arguments
    -----------
    x       :   np.ndarray
                flat vector of values to compute the effective sample size

    use_R   :   bool
                option to use rpy2+CODA or pure python implementation. If False,
                the effective size computation may be unbearably slow on large data,
                due to slow AR fitting in statsmodels.
    """
    if use_R:
        if _HAS_RPY2 and _HAS_CODA:
            return _coda.effectiveSize(x)[0]
        elif _HAS_RPY2 and not _HAS_CODA:
            raise ImportError("No module named 'coda' in R.")
        else:
            raise ImportError("No module named 'rpy2'")
    else:
        spec = _spectrum0_ar(x)
        if spec == 0:
            return 0
        loss_factor = _np.var(x, ddof=1)/spec
        return len(x) * loss_factor

#############################
# Highest Posterior Density #
#############################

def hpd_interval(model = None,  trace = None,  chain = None,  varnames = None,  p=.95):
    """

    Parameters
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples
                of p covariates.
    varnames:   str or list of str
                set of variates to extract from the model or trace to to compute the
                statistic.
    p       :   float
                percent of highest density to extract

    Returns
    -------
    hashmap of results, where each result is {'varname':(low, hi)}

    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    stats = trace.map(_hpd_interval, p=p)
    return stats if len(stats) > 1 else stats[0]

def _hpd_interval(data, p=.95):
    """

    Parameters
    ----------
    data    :   numpy.ndarray
                data to compute the hpd
    p       :   float
                percent of highest density to extract

    Returns
    -------
    tuple of (low,hi) boundaries of the highest posterior fraction.

    """
    data = _np.sort(data)
    N = len(data)
    N_in = int(_np.ceil(N*p))
    head = _np.arange(0,N-N_in)
    tail = head+N_in
    pivot = _np.argmin(data[tail] - data[head])
    return data[pivot], data[pivot+N_in]

def point_estimates(model=None, trace=None, chain=None, 
                    burnin=0,thin=1,
                    varnames=None, statistic=_np.median):
    """
    Get a point estimate for the posterior
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    trace = _resolve_to_trace(None, trace[burnin::thin], None, varnames)
    return trace.map(statistic)


############################################
# Markov Chain Monte Carlo Standard Errors #
############################################

def mcse(model = None, trace=None, chain = None, varnames = None,
           rescale=2, method='bartlett', N_chunks=None, transform=_thru_op):
    """

    Parameters
    ----------
    model   :   Any model object.
                must have an attached `trace` attribute. Takes precedence over
                `trace` and `chain` arguments.
    trace   :   abstracts.Trace
                a trace object containing data to compute the diagnostic
    chain   :   np.ndarray
                an array indexed by (m,n[,p]) containing m parallel runs of n samples
                of p covariates.
    varnames:   str or list of str
                set of variates to extract from the model or trace to to compute the
                statistic.
    rescale :   real positive number
                governs the shrinkage/reduction of the relevant data
    method  :   str (default: 'bartlett')
                string describing method used to compute the standard errors.
                Supported options:
                    - 'bm' : batch means, reduces chain by the mean of each `N_chunks` block
                    - 'obm': overlapping batch means, reduces chain by the mean of `N_chunks` rolling blocks
                    - 'tukey': weighted reduction using a Tukey window (see np.tukey)
                    - 'hanning': same as 'tukey'
                    - 'bartlett': weighted reduction using a Bartlett window (see np.bartlett)
    transform:  callable
                function or callable class that consumes data and returns it in an identical shape. Used
                if the values of interest are a transformation of the given parameter.
    Returns
    -------
    hashmap or list of hashmaps that contain the standard errors of the given chain.
    """
    trace = _resolve_to_trace(model, trace, chain, varnames)
    out = trace.map(_mcse, rescale=rescale, method=method, N_chunks=N_chunks, transform=transform)
    return out if len(out) > 1 else out[0]

def _mcse(x, rescale=2, N_chunks=None, method='bm', transform=_thru_op):
    if rescale is None and N_chunks is None:
        raise ValueError("Either 'rescale' or 'N_chunks' must be supplied.")
    elif rescale is not None and N_chunks is None:
        size = len(x)
        if 0 < rescale < 1:
            rescale = 1/rescale
        chunk_size = _np.floor(size**(1.0/rescale))
        N_chunks = _np.floor(size / chunk_size)
    elif N_chunks is not None:
        pass
    else:
        raise Exception("Options 'rescale' and 'N_chunks' were not resolved successfully!")

    try:
        method = _mcse_dispatch[method]
    except KeyError:
        raise KeyError("Supported methods are: 'bm', 'obm', 'bartlett', 'tukey'")
    return method(x, int(N_chunks), transform = transform)

def _mcse_bm(x, N_chunks, transform = _thru_op):
    """
    Compute a Markov Chain Monte Carlo Standard Error
     using raw batch means

    Parameters
    ----------
    x           :   numpy.ndarray
    N_chunks    :   int
    transform   :   callable

    Returns
    -------
    float containing the standard error of x
    """
    N = len(x)
    chunk_size = _np.floor(N / N_chunks).astype(int)
    y = _np.asarray([transform(split).mean() for split in _np.array_split(x, N_chunks)])
    mean = transform(x).mean()
    variance = chunk_size * ((y - mean)**2).sum() / (N_chunks - 1) #isn' this chunk_size * (x.var(ddof=1)?)
    return _np.sqrt(variance / N)

def _mcse_obm(x, N_chunks, transform = _thru_op):
    """
    Compute a Markov Chain Monte Carlo Standard Error
     using overlapping batch means

    Parameters
    ----------
    x           :   numpy.ndarray
    N_chunks    :   int
    transform   :   callable

    Returns
    -------
    float containing the standard error of x
    """
    N = len(x)
    a = N - N_chunks + 1
    chunk_size = _np.floor(N / N_chunks).astype(int)
    y = _pd.Series(x).rolling(chunk_size).apply(lambda vec: _np.mean(transform(vec)))
    y = y[~_np.isnan(y)]
    mean = transform(x).mean()
    variance = N * chunk_size * ((y - mean)**2).sum() / (a -1) / a
    return _np.sqrt(variance / N)

def _mcse_bartlett(x, N_chunks, transform = _thru_op):
    """
    Compute a Markov Chain Monte Carlo Standard Error
     using a Bartlett window

    Parameters
    ----------
    x           :   numpy.ndarray
    N_chunks    :   int
    transform   :   callable

    Returns
    -------
    float containing the standard error of x
    """
    N = len(x)
    chunk_size = _np.floor(N / N_chunks).astype(int)
    brange = _np.arange(1, chunk_size+1)
    alpha = (1 - brange / chunk_size) * (1 - brange / N)
    mean = transform(x).mean()
    diffs = ((x[0:(N-i)] - mean) * (x[i:N] - mean) for i in range(chunk_size+1))
    R = _np.asarray([diff.mean() for diff in diffs])
    variance = R[0] + 2 * (alpha * R[1:]).sum()
    return _np.sqrt(variance / N)

def _mcse_hanning(x, N_chunks, transform = _thru_op):
    """
    Compute a Markov Chain Monte Carlo Standard Error
     using a Tukey window

    Parameters
    ----------
    x           :   numpy.ndarray
    N_chunks    :   int
    transform   :   callable

    Returns
    -------
    float containing the standard error of x
    """
    N = len(x)
    chunk_size = _np.floor(N / N_chunks).astype(int)
    brange = _np.arange(1, chunk_size+1)
    alpha = (1 + _np.cos(_np.pi * brange / chunk_size)) / 2 * (1 - brange / N)
    mean = transform(x).mean()
    diffs = ((x[0:(N-i)] - mean) * (x[i:N] - mean) for i in range(chunk_size+1))
    R = _np.asarray([diff.mean() for diff in diffs])
    variance = R[0] + 2 * (alpha * R[1:]).sum()
    return _np.sqrt(variance / N)

_mcse_dispatch = dict(
    tukey = _mcse_hanning,
    hanning = _mcse_hanning,
    bartlett = _mcse_bartlett,
    obm = _mcse_obm,
    bm = _mcse_bm
)

#############
# Utilities #
#############

def _resolve_to_trace(model, trace, chain, varnames):
    """
    Resolve a collection of information down to a trace. This reduces the
    passed arguments to a trace that can be used for analysis based on names
    in varnames.

    If `trace` is passed, it is subset according to `varnames`, and a copy returned.
    It takes precedence.
    Otherwise, if `model` is passed, its traces are taken.
    Finally, if `chain` is passed, a trace is constructed to structure the chain.
    In all cases, if `varnames` is passed, it is used to name or subset the given data.

    """
    n_passed = sum([model is not None, trace is not None, chain is not None])
    if n_passed > 1:
        raise Exception('Only one of `model`, `trace`, or `chain` '
                        ' may be passed.')
    if isinstance(varnames, str):
        varnames = [varnames]
    if trace is not None:
        if isinstance(trace, _Trace):
            if varnames is not None:
                return trace.drop([var for var in trace.varnames
                                    if var not in varnames], inplace=False)
            else:
                return _copy.deepcopy(trace)
        else:
            return _Trace(*_copy_hashmaps(_maybe_hashmap(trace)[0]))

    if model is not None:
        return _resolve_to_trace(model=None, trace=model.trace,
                                 chain=None, varnames=varnames)
    if chain is not None:
        if chain.ndim > 2:
            m, n = chain.shape[0:2]
            rest = chain.shape[2:]
            new_p = _np.multiply(*rest)
            chain = chain.reshape(m, n, new_p)
        elif chain.ndim == 1:
            if varnames is None:
                varnames = ['parameter_0']
            elif isinstance(varnames, list):
                varnames = varnames[0]
            return _Trace(_Hashmap(**dict([(varnames[0], chain)])))
        if varnames is None:
            varnames = ['parameter_{}'.format(i) for i in range(len(chain))]
        else:
            if len(varnames) != new_p:
                raise NotImplementedError('Parameter Subsetting by varnames '
                                          'is not currenlty implented for raw arrays')
        return _Trace(*[_Hashmap(**{k: run.T[p] for p, k in enumerate(varnames)})
                      for run in chain])
