"""
Canned Views using PySAL and Matplotlib
"""

__author__ = "Marynia Kolak <marynia.kolak@gmail.com>"

import pandas as pd
import numpy as np
import pysal as ps
import matplotlib.pyplot as plt


__all__ = ['mplot']


def mplot(m, xlabel='', ylabel='', title='', custom=(7,7)):
    """
    Produce basic Moran Plot 

    Parameters
    ----------
    m : pysal.Moran instance
        values of Moran's I Global Autocorrelation Statistic
    xlabel : str
        label for x axis
    ylabel : str
        label for y axis
    title : str
        title of plot
    custom : tuple
        dimensions of figure size

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import pysal as ps
    >>> from pysal.contrib.pdio import read_files
    >>> from pysal.contrib.viz.plot import mplot

    >>> link = ps.examples.get_path('columbus.shp')
    >>> db = read_files(link)
    >>> y = db['HOVAL'].values
    >>> w = ps.queen_from_shapefile(link)
    >>> w.transform = 'R'

    >>> m = ps.Moran(y, w)
    >>> mplot(m, xlabel='Response', ylabel='Spatial Lag',
    ...       title='Moran Scatterplot', custom=(7,7))

    >>> plt.show()
            
    """
    lag = ps.lag_spatial(m.w, m.z)
    fit = ps.spreg.OLS(m.z[:, None], lag[:,None])

    # Customize plot
    fig = plt.figure(figsize=custom)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    ax.scatter(m.z, lag, s=60, color='k', alpha=.6)
    ax.plot(lag, fit.predy, color='r')

    ax.axvline(0, alpha=0.5)
    ax.axhline(0, alpha=0.5)

    return fig
