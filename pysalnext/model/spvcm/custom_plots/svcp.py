import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    return f,ax
