import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(group_share1, group_share2, label1='', label2=''):
    """Plot CDF for two series.

    Convenience function for comparing inequality between two series by
    plotting their CDFs on the same graph

    Parameters
    ----------
    group_share1 : pd.Series
        pandas series with variable of interest.
    group_share2 : pd.Series
        pandas series with variable of interest.
    label1 : str
        legend label for first series
    label2: str
        legend label for second series

    Returns
    -------
    type
        matplotlib Figure.

    """
    plt.step(group_share1.sort_values(),
             group_share1.rank(pct=True).sort_values(),
             label=label1)
    plt.step(group_share2.sort_values(),
             group_share2.rank(pct=True).sort_values(),
             label=label2)
    if (label1 != '' or label2 != ''):
        plt.legend()
    plt.show()
    fig = plt.gcf()
    return fig


def lorenz(X):
    """Plot lorenx curve.

    Parameters
    ----------
    X : pandas.Series
        series of values to plot.

    Returns
    -------
    type
        matplotlib.Figure

    """

    fig, ax = plt.subplots(figsize=[6, 6])

    X = X.sort_values()
    X_lorenz = X.cumsum() / X.sum()
    ax.scatter(np.arange(X_lorenz.size) / (X_lorenz.size - 1),
               X_lorenz,
               marker='x',
               color='navy',
               s=10)
    ax.plot([0, 1], [0, 1], color='k', linewidth=1)
    return ax
