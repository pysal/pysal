import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['plot_trace', 'seplot', 'rollplot', 'corrplot']

def plot_trace(model, burn=0, thin=None, varnames=None, trace=None,
               kde_kwargs={}, trace_kwargs={}, figure_kwargs={}):
    """
    Make a trace plot paired with a distributional plot.

    Arguments
    -----------
    model   :   Model object
                a model with a trace attribute
    burn    :   int
                the number of iterations to discard from the front of the trace.
                If negative, the number of iterations to use from the tail of the trace.
    thin    :   int
                the number of iterations to discard between iterations
    varnames :  str or list
                name or list of names to plot.
    trace   :   namespace
                a namespace whose variables are contained in varnames
    kde_kwargs : dictionary
                 dictionary of aesthetic arguments for the kde plot
    trace_kwargs : dictionary
                   dictinoary of aesthetic arguments for the traceplot
    figure_kwargs: dictionary
                    a dictionary of arguments for the plot creator

    Returns
    -------
    figure, axis tuple, where axis is (len(varnames), 2)
    """
    if thin is None or thin is 0:
        thin = 1
    if model is None:
        if trace is None:
            raise Exception('Neither model nor trace provided.')
    else:
        trace = model.trace
    if varnames is None:
        varnames = trace.varnames
    elif isinstance(varnames, str):
        varnames = [varnames]
    if figure_kwargs == dict():
        figure_kwargs = {'figsize':(8, 2*len(varnames)), 'sharey':'row'}
    if kde_kwargs == dict():
        kde_kwargs = {'shade':True, 'vertical':True}
    if trace_kwargs == dict():
        trace_kwargs = {'linewidth':.5}
    fig, ax = plt.subplots(len(varnames), 2, **figure_kwargs)
    for chain_i, chain in enumerate(trace.chains):
        for i, param_name in enumerate(varnames):
            this_param = np.asarray(trace[chain_i,param_name,burn::thin])
            if ax.ndim ==1:
                ax = np.array((ax,),)
            if len(this_param.shape) == 3:
                n,a,b = this_param.shape
                this_param = this_param.reshape(n,a*b)
            if len(this_param.shape) == 2:
                if this_param.shape[-1] == 1:
                    sns.kdeplot(this_param.flatten(),
                                ax=ax[i,1], **kde_kwargs)
                else:
                    for param in this_param.T:
                        sns.kdeplot(param, ax=ax[i,1], **kde_kwargs)
            else:
                sns.kdeplot(this_param, ax=ax[i,1], **kde_kwargs)
            ax[i,0].plot(this_param, **trace_kwargs)
            ax[i,1].set_title(param_name)
            ax[i,0].set_xbound(0, len(this_param))
    fig.tight_layout()
    return fig, ax

def seplot(model=None, trace=None, chain=None, varnames=None,
           burn=0, thin=None, N_bins=200,
           plot_kw=None, fig_kw=None, ax=None):
    """
    This plots the markov chain monte carlo standard error 
    as a function of iterations. 

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
    N_bins  :   int
                number of subsamples over which to compute the standard error
    roller  :   callable
                pandas rolling function that takes argument `window` to specify window size
    burn    :   int
                number of iterations to discard from the front of the chain. If negative, number of iterations to keep from the tail of the chain.
    thin    :   int
                step to thin the chain. Picks every `thin`th observation.
    plot_kw :   dict/keyword arguments
                passed to the line plot call
    fig_kw  :   dict/keyword arguments
                passed to the plt.subplots() call
    ax      :   matplotlib axis
                pass if you want to plot this on an existing set of axes.

    Returns
    --------
    figure,axis tuple or, if ax is passed, ax
    """
    from . import diagnostics as diag
    trace = diag._resolve_to_trace(model, trace, chain, varnames)

    thin = 1 if thin is None else thin

    trace = diag._resolve_to_trace(None, trace[burn::thin], None, None)
    statistics = trace.map(_se_vector, N_bins=N_bins)

    if fig_kw is None:
        fig_kw = dict()
    fig_kw['figsize'] = fig_kw.get('figsize', (5, 2*len(trace.varnames)))
    fig_kw['sharex'] = fig_kw.get('sharex', True)
    if plot_kw is None:
        plot_kw = dict()

    if ax is None:
        f,ax = plt.subplots(len(trace.varnames), 1, **fig_kw)

    for i, varname in enumerate(trace.varnames):
        to_plot = (np.asarray(result[varname]) for result in statistics)
        for results in to_plot:
            if results.ndim > 1:
                results = results.T
            ax[i].plot(results, **plot_kw)
            ax[i].set_title(varname)
        ax[i].set_ylabel('MC Standard Error')
    try:
        f.tight_layout()
        return f,ax
    except NameError:
        return ax


def rollplot(model=None, trace=None, chain=None, varnames=None,
             order=100, roller=None,
             burn=0, thin=None, plot_kw=None, fig_kw=None, ax=None):
    """
    This plots a rolling window function, `roller`, against all parameters contained
    in the model, trace, or chain. 

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
    order   :   int
                the order of the moving process
    roller  :   callable
                function that takes arguments `data, window`, where `data` is all data and `window` is the size
                of the window to pass over `data`, applying function `roller`. 
    burn    :   int
                number of iterations to discard from the front of the chain. If negative, number of iterations to keep from the tail of the chain.
    thin    :   int
                step to thin the chain. Picks every `thin`th observation.
    plot_kw :   dict/keyword arguments
                passed to the line plot call
    fig_kw  :   dict/keyword arguments
                passed to the plt.subplots() call
    ax      :   matplotlib axis
                pass if you want to plot this on an existing set of axes.

    Returns
    --------
    figure,axis tuple or, if ax is passed, ax
    """
    from . import diagnostics as diag
    if roller is None:
        roller = lambda data, order: pd.Series(data).rolling(order).mean()

    trace = diag._resolve_to_trace(model, trace, chain, varnames)

    thin = 1 if thin is None else thin
    trace = diag._resolve_to_trace(model, trace[burn::thin], chain, varnames)

    if fig_kw is None:
        fig_kw = dict()
    fig_kw['figsize'] = fig_kw.get('figsize', (5, 2*len(trace.varnames)))
    fig_kw['sharex'] = fig_kw.get('sharex', True)
    if plot_kw is None:
        plot_kw = dict()

    if ax is None:
        f,ax = plt.subplots(len(trace.varnames), 1, **fig_kw)
    for i, varname in enumerate(trace.varnames):
        for j in range(trace.n_chains):
            data = trace[j, varname]
            ax[i].plot(roller(data, order), **plot_kw)
        ax[i].set_title(varname)
    try:
        f.tight_layout()
        return f,ax
    except NameError:
        return ax


def conv_plot(model=None, trace=None, chain=None, varnames=None,
              N_bins=200, roller=None,
              burn=0, thin=None, plot_kw=None, fig_kw=None, ax=None):
    """
    This plots a moving average of the chain alongside of a standard error indicator. This takes a long time to compute and provides poor plots unless N_bins is very high relative to the chain length.

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
    N_bins  :   int
                the number of sample points to use to compute the rolling mean and standard errors.
    roller  :   callable
                function that takes arguments `data, window`, where `data` is all data and `window` is the size
                of the window to pass over `data`, applying function `roller`. 
    burn    :   int
                number of iterations to discard from the front of the chain. If negative, number of iterations to keep from the tail of the chain.
    thin    :   int
                step to thin the chain. Picks every `thin`th observation.
    plot_kw :   dict/keyword arguments
                passed to the line plot call
    fig_kw  :   dict/keyword arguments
                passed to the plt.subplots() call
    ax      :   matplotlib axis
                pass if you want to plot this on an existing set of axes.

    Returns
    --------
    figure,axis tuple or, if ax is passed, ax

    """
    from . import diagnostics as diag
    if roller is None:
        roller = lambda data, order: pd.Series(data).rolling(order).mean()

    trace = diag._resolve_to_trace(model, trace, chain, varnames)

    thin = 1 if thin is None else thin

    trace = diag._resolve_to_trace(None, trace[burn::thin], None,None)

    if fig_kw is None:
        fig_kw = dict()
    fig_kw['figsize'] = fig_kw.get('figsize', (5, 2*len(trace.varnames)))
    fig_kw['sharex'] = fig_kw.get('sharex', 'col')
    if plot_kw is None:
        plot_kw = dict()

    f,ax = plt.subplots(len(trace.varnames), 1, **fig_kw)

    window = trace.n_iters - N_bins + 1
    rolled = trace.map(roller, window=window)
    ses = trace.map(_se_vector, N_bins=N_bins)

    for i, varname in enumerate(trace.varnames):
        for j in range(trace.n_chains):
            this_roll = np.asarray(rolled[j][varname])
            if this_roll.ndim > 1:
                this_roll = this_roll[~np.isnan(this_roll)].reshape(-1, N_bins)
            else:
                this_roll = this_roll[~np.isnan(this_roll)]
            this_se = np.asarray(ses[j][varname])

            upper = this_roll + this_se * 2
            lower = this_roll - this_se * 2
            if this_roll.ndim > 1:
                for this, up, low in zip(this_roll, upper, lower):
                    ax[i].fill_between(np.arange(len(this)), up, low, where = up>low, alpha=.5)
            else:
                ax[i].fill_between(np.arange(len(this_roll)), upper, lower, where = upper>lower, alpha=.5)
            ax[i].plot(this_roll, **plot_kw)
        ax[i].set_title(varname)
    return f,ax


def _se_vector(x, N_bins):
    """
    A vectorized version of the monte carlo standard error estimator. Computes the
    monte carlo standard error by breaking `x` into `N_bins`, and then compuing the standard error over the chain defined by concatenating the bins starting at the head of the chain.

    If the chain is converged, the standard error estimator should converge to zero in this procedure. The speed of convergence is indicated by how quickly the standard error reduces.

    Arguments
    ----------
    x       :   numpy.ndarray
                an array over which to compute the standard errors
    N_bins  :   int
                number of bins to break `x` into to compute standard errors

    Returns
    ----------
    N_bins-length vector containing whose element i contains the standard error of the x[:i+1] chunk.
    """
    from . import diagnostics as diag

    splits = np.array_split(x, N_bins)
    diags = [list(diag.mcse(chain=np.hstack(splits[:i+1])).values())[0] for i in range(N_bins)]
    return diags

def corrplot(m, burn=0, thin=None,
             percentiles=[25,50,75], support=np.linspace(.001,1,num=1000),
             figure_kw=None, plot_kw=None, kde_kw=None):
    if figure_kw is None:
        figure_kw = {'figsize':(2.1*8,8), 'sharey':True}

    if plot_kw is None:
        plot_kw = [dict()]*len(percentiles)
    elif isinstance(plot_kw, dict):
        plot_kw = [plot_kw]*len(percentiles)
    elif isinstance(plot_kw, list):
        assert len(plot_kw)==len(percentiles)

    if kde_kw is None:
        kde_kw = [{'vertical':True, 'shade':True}]*len(percentiles)
    elif isinstance(kde_kw, dict):
        kde_kw = [kde_kw]*len(percentiles)
    elif isinstance(kde_kw, list):
        assert len(kde_kw)==len(percentiles)

    corrfunc = m.state.correlation_function
    pwds = m.state.pwds
    if m.trace.n_chains > 1:
        raise
    phis = m.trace['Phi', burn::thin]
    f,ax = plt.subplots(1,2, **figure_kw)
    support = np.linspace(.001,1,num=1000)
    ptiles = [[np.percentile(corrfunc(r, pwds).flatten(), ptile)
               for r in support] for ptile in percentiles]
    empirical_median_correlation = [np.median(corrfunc(phi, pwds)) for phi in phis]
    for i, ptile in enumerate(ptiles):
        ax[0].plot(support*m.state.max_dist, ptile, **plot_kw[i])
    sns.kdeplot(np.asarray(empirical_median_correlation), ax=ax[1], **kde_kw[i])
    ax[0].set_title('Percentile of Correlation')
    ax[1].set_title('Median spatial correlations')
    ax[1].set
    ax[0].set_ybound(0,1)
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Inter-Observation $\\rho$')
    return f, ax


def hpd_trajplot(model=None, trace=None, chain=None, varnames=None,
                 alpha=.95, n_splits=100,
                 fig_kw=dict(), hpdi_kw=dict(), trace_kw=dict(), width_kw=dict()):
    """
    This plots the middle alpha% interval around the parameter over n_splits blocks. 

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
    alpha   :   float
                the percentile to use for the HPD interval plotted in each chunk
    n_splits:   int
                the number of chunks of the chain to estimate the HPD over. 
    burn    :   int
                number of iterations to discard from the front of the chain. If negative,
                number of iterations to keep from the tail of the chain.
    thin    :   int
                step to thin the chain. Picks every `thin`th observation.
    fig_kw  :   dict/keyword arguments
                passed to the plt.subplots() call
    hpdi_kw :   dict/keyword arguments
                passed to the diagnostics.hpd_interval() call for the hpd envelope
    trace_kw:   dict/keyword arguments
                passed to the plt.plot() call for the trace of the parameters
    width_kw:   dict/keyword arguments
                passed to the plt.plot() call for the intervals of parameters
    ax      :   matplotlib axis
                pass if you want to plot this on an existing set of axes.

    Returns
    --------
    figure,axis tuple or, if ax is passed, ax

    """
    from . import diagnostics as diag
    trace = diag._resolve_to_trace(model, trace, chain, varnames)
    if varnames is None:
        varnames = trace.varnames
    p = len(varnames)
    f, ax = plt.subplots(p, 2, **fig_kw)
    pieces = trace.map(np.array_split, indices_or_sections=n_splits)
    hpds = dict()
    for i, varname in enumerate(varnames):
        bits = pieces[varname]
        hpds[varname] = []
        for i, bit in enumerate(bits):
            try:
                cumulant = np.hstack((cumulant, bit))
                hpd = diag.hpd_interval(chain=cumulant, alpha=alpha)
            except NameError:
                cumulant = bit
                hpd = diag.hpd_interval(chain=cumulant, alpha=alpha)
            finally:
                hpds[varname].append(hpd)
        this_hpdset = np.hstack(hpds[varname]).reshape(-1, 2)
        keff = len(this_hpdset)
        support = np.arange(0, keff)
        i, ax[0].plot(support, this_hpdset.T[0], **hpdi_kw)
        i, ax[0].plot(support, this_hpdset.T[1], **hpdi_kw)
        i, ax[0].plot(trace[varname], **trace_kw)
        widths = np.subtract.reduce(this_hpdset, axis=1)
        i, ax[1].plot(support, widths, **width_kw)
    return f, ax
