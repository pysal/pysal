import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import numpy as np
from pysal.lib.weights.contiguity import Queen
from pysal.lib.weights.spatial_lag import lag_spatial
import seaborn as sbn
from pysal.explore.esda.moran import (Moran_Local, Moran_Local_BV,
                        Moran, Moran_BV)
import warnings
from pysal.model.spreg import OLS

from matplotlib import patches, colors

from ._viz_utils import (mask_local_auto, moran_hot_cold_spots,
                         splot_colors)

"""
Lightweight visualizations for esda using Matplotlib and Geopandas

TODO
* geopandas plotting, change round shapes in legends to boxes
* prototype moran_facet using `seaborn.FacetGrid`
"""

__author__ = ("Stefanie Lumnitz <stefanie.lumitz@gmail.com>")


def _create_moran_fig_ax(ax, figsize, aspect_equal):
    """
    Creates matplotlib figure and axes instances
    for plotting moran visualizations. Adds common viz design.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    
    ax.spines['left'].set_position(('axes', -0.05))
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['top'].set_color('none')
    if aspect_equal is True:
        ax.set_aspect('equal')
    else:
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
    return fig, ax


def moran_scatterplot(moran, zstandard=True, p=None,
                      aspect_equal=True, ax=None,
                      scatter_kwds=None, fitline_kwds=None):
    """
    Moran Scatterplot
    
    Parameters
    ----------
    moran : esda.moran instance
        Values of Moran's I Global, Bivariate and Local
        Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    p : float, optional
        If given, the p-value threshold for significance
        for Local Autocorrelation analysis. Points will be colored by
        significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions
        for Moran Scatterplot.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted
    
    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import (Moran, Moran_BV,
    ...                         Moran_Local, Moran_Local_BV)
    >>> from pysal.viz.splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate esda.moran Objects
    
    >>> moran = Moran(y, w)
    >>> moran_bv = Moran_BV(y, x, w)
    >>> moran_loc = Moran_Local(y, w)
    >>> moran_loc_bv = Moran_Local_BV(y, x, w)
    
    Plot
    
    >>> fig, axs = plt.subplots(2, 2, figsize=(10,10),
    ...                         subplot_kw={'aspect': 'equal'})
    >>> moran_scatterplot(moran, p=0.05, ax=axs[0,0])
    >>> moran_scatterplot(moran_loc, p=0.05, ax=axs[1,0])
    >>> moran_scatterplot(moran_bv, p=0.05, ax=axs[0,1])
    >>> moran_scatterplot(moran_loc_bv, p=0.05, ax=axs[1,1])
    >>> plt.show()
    
    """
    if isinstance(moran, Moran):
        if p is not None:
            warnings.warn('`p` is only used for plotting `esda.moran.Moran_Local`\n'
                          'or `Moran_Local_BV` objects')
        fig, ax = _moran_global_scatterplot(moran=moran, zstandard=zstandard,
                                            ax=ax, aspect_equal=aspect_equal,
                                            scatter_kwds=scatter_kwds,
                                            fitline_kwds=fitline_kwds)
    elif isinstance(moran, Moran_BV):
        if p is not None:
            warnings.warn('`p` is only used for plotting `esda.moran.Moran_Local`\n'
                          'or `Moran_Local_BV` objects')
        fig, ax = _moran_bv_scatterplot(moran_bv=moran, ax=ax,
                                        aspect_equal=aspect_equal,
                                        scatter_kwds=scatter_kwds,
                                        fitline_kwds=fitline_kwds)
    elif isinstance(moran, Moran_Local):
        fig, ax = _moran_loc_scatterplot(moran_loc=moran, zstandard=zstandard,
                                         ax=ax, p=p, aspect_equal=aspect_equal,
                                         scatter_kwds=scatter_kwds,
                                         fitline_kwds=fitline_kwds)
    elif isinstance(moran, Moran_Local_BV):
        fig, ax = _moran_loc_bv_scatterplot(moran_loc_bv=moran, ax=ax,
                                            p=p, aspect_equal=aspect_equal,
                                            scatter_kwds=scatter_kwds,
                                            fitline_kwds=fitline_kwds)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return fig, ax


def _moran_global_scatterplot(moran, zstandard=True,
                              aspect_equal=True, ax=None,
                              scatter_kwds=None, fitline_kwds=None):
    """
    Global Moran's I Scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes will show the same aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran
    >>> from pysal.viz.splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> moran_scatterplot(moran)
    >>> plt.show()
    
    customize plot
    
    >>> fig, ax = moran_scatterplot(moran, zstandard=False,
    ...                             fitline_kwds=dict(color='#4393c3'))
    >>> ax.set_xlabel('Donations')
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    # define customization defaults
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('color', splot_colors['moran_base'])
    scatter_kwds.setdefault('s', 40)
    
    fitline_kwds.setdefault('alpha', 0.9)
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    
    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7, 7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Spatial Lag')
    ax.set_title('Moran Scatterplot' +
                 ' (' + str(round(moran.I, 2)) + ')')

    # plot and set standards
    if zstandard is True:
        lag = lag_spatial(moran.w, moran.z)
        fit = OLS(moran.z[:, None], lag[:, None])
        # plot
        ax.scatter(moran.z, lag, **scatter_kwds)
        ax.plot(lag, fit.predy, **fitline_kwds)
        # v- and hlines
        ax.axvline(0, alpha=0.5, color='k', linestyle='--')
        ax.axhline(0, alpha=0.5, color='k', linestyle='--')
    else:
        lag = lag_spatial(moran.w, moran.y)
        b, a = np.polyfit(moran.y, lag, 1)
        # plot
        ax.scatter(moran.y, lag, **scatter_kwds)
        ax.plot(moran.y, a + b*moran.y, **fitline_kwds)
        # dashed vert at mean of the attribute
        ax.vlines(moran.y.mean(), lag.min(), lag.max(), alpha=0.5,
                  linestyle='--')
        # dashed horizontal at mean of lagged attribute
        ax.hlines(lag.mean(), moran.y.min(), moran.y.max(), alpha=0.5,
                  linestyle='--')
    return fig, ax


def plot_moran_simulation(moran, aspect_equal=True,
                          ax=None, fitline_kwds=None,
                          **kwargs):
    """
    Global Moran's I simulated reference distribution.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborn.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Simulated reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran
    >>> from pysal.viz.splot.esda import plot_moran_simulation
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> plot_moran_simulation(moran)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_simulation(moran, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop('figsize', (7, 7))
    
    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize,
                                   aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop('shade', True)
    color = kwargs.pop('color', splot_colors['moran_base'])
    sbn.kdeplot(moran.sim, shade=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    ax.vlines(moran.I, 0, 1, **fitline_kwds)
    ax.vlines(moran.EI, 0, 1)
    ax.set_title('Reference Distribution')
    ax.set_xlabel('Moran I: ' + str(round(moran.I, 2)))
    return fig, ax


def plot_moran(moran, zstandard=True, aspect_equal=True,
               scatter_kwds=None, fitline_kwds=None, **kwargs):
    """
    Global Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran : esda.moran.Moran instance
        Values of Moran's I Global Autocorrelation Statistics
    zstandard : bool, optional
        If True, Moran Scatterplot will show z-standardized attribute and
        spatial lag values. Default =True.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran
    >>> from pysal.viz.splot.esda import plot_moran
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Global Moran
    
    >>> moran = Moran(y, w)
    
    plot
    
    >>> plot_moran(moran)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran(moran, zstandard=False,
    ...            fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    figsize = kwargs.pop('figsize', (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            subplot_kw={'aspect': 'equal'})
    plot_moran_simulation(moran, ax=axs[0], fitline_kwds=fitline_kwds, **kwargs)
    moran_scatterplot(moran, zstandard=zstandard, ax=axs[1],
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds)
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else: 
        axs[1].set_aspect("auto")
    return fig, axs


def _moran_bv_scatterplot(moran_bv, ax=None, aspect_equal=True,
                          scatter_kwds=None, fitline_kwds=None):
    """
    Bivariate Moran Scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_BV
    >>> from pysal.viz.splot.esda import moran_scatterplot
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> moran_scatterplot(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> moran_scatterplot(moran_bv,
    ...                      fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    # define customization
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('color', splot_colors['moran_base'])
    scatter_kwds.setdefault('s', 40)
    
    fitline_kwds.setdefault('alpha', 0.9)
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7,7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute X')
    ax.set_ylabel('Spatial Lag of Y')
    ax.set_title('Bivariate Moran Scatterplot' +
                 ' (' + str(round(moran_bv.I, 2)) + ')')

    # plot and set standards
    lag = lag_spatial(moran_bv.w, moran_bv.zy)
    fit = OLS(moran_bv.zy[:, None], lag[:, None])
    # plot
    ax.scatter(moran_bv.zx, lag, **scatter_kwds)
    ax.plot(lag, fit.predy, **fitline_kwds)
    # v- and hlines
    ax.axvline(0, alpha=0.5, color='k', linestyle='--')
    ax.axhline(0, alpha=0.5, color='k', linestyle='--')
    return fig, ax


def plot_moran_bv_simulation(moran_bv, ax=None, aspect_equal=True,
                             fitline_kwds=None, **kwargs):
    """
    Bivariate Moran's I simulated reference distribution.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the
        vertical moran fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_BV
    >>> from pysal.viz.splot.esda import plot_moran_bv_simulation
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> plot_moran_bv_simulation(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_bv_simulation(moran_bv,
        ...                      fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if fitline_kwds is None:
        fitline_kwds = dict()

    figsize = kwargs.pop('figsize', (7, 7))

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize,
                                   aspect_equal=aspect_equal)

    # plot distribution
    shade = kwargs.pop('shade', True)
    color = kwargs.pop('color', splot_colors['moran_base'])
    sbn.kdeplot(moran_bv.sim, shade=shade, color=color, ax=ax, **kwargs)

    # customize plot
    fitline_kwds.setdefault('color', splot_colors['moran_fit'])
    ax.vlines(moran_bv.I, 0, 1, **fitline_kwds)
    ax.vlines(moran_bv.EI_sim, 0, 1)
    ax.set_title('Reference Distribution')
    ax.set_xlabel('Bivariate Moran I: ' + str(round(moran_bv.I, 2)))
    return fig, ax


def plot_moran_bv(moran_bv, aspect_equal=True,
                  scatter_kwds=None, fitline_kwds=None, **kwargs):
    """
    Bivariate Moran's I simulated reference distribution and scatterplot.

    Parameters
    ----------
    moran_bv : esda.moran.Moran_BV instance
        Values of Bivariate Moran's I Autocorrelation Statistics
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        and vertical fitline. Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the figure,
        passed to seaborne.kdeplot.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate moran scatterplot and reference distribution figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_BV
    >>> from pysal.viz.splot.esda import plot_moran_bv
    
    Load data and calculate weights
    
    >>> link_to_data = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link_to_data)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    
    Calculate Bivariate Moran
    
    >>> moran_bv = Moran_BV(x, y, w)
    
    plot
    
    >>> plot_moran_bv(moran_bv)
    >>> plt.show()
    
    customize plot
    
    >>> plot_moran_bv(moran_bv, fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    figsize = kwargs.pop('figsize', (10, 4))
    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            subplot_kw={'aspect': 'equal'})
    plot_moran_bv_simulation(moran_bv, ax=axs[0], fitline_kwds=fitline_kwds,
                             **kwargs)
    moran_scatterplot(moran_bv, ax=axs[1], aspect_equal=aspect_equal,
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds)
    axs[0].set(aspect="auto")
    if aspect_equal is True:
        axs[1].set_aspect("equal", "datalim")
    else:
        axs[1].set(aspect="auto")
    return fig, axs


def _moran_loc_scatterplot(moran_loc, zstandard=True, p=None, aspect_equal=True,
                           ax=None, scatter_kwds=None, fitline_kwds=None):
    """
    Moran Scatterplot with option of coloring of Local Moran Statistics

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local instance
        Values of Moran's I Local Autocorrelation Statistics
    p : float, optional
        If given, the p-value threshold for significance. Points will
        be colored by significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Moran Local scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> import geopandas as gpd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> from pysal.explore.esda.moran import Moran_Local
    >>> from pysal.viz.splot.esda import moran_scatterplot
    
    Load data and calculate Moran Local statistics
    
    >>> link = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> m = Moran_Local(y, w)
    
    plot
    
    >>> moran_scatterplot(m)
    >>> plt.show()
    
    customize plot
    
    >>> moran_scatterplot(m, p=0.05,
    ...                   fitline_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    if p is not None:
        if not isinstance(moran_loc, Moran_Local):
            raise ValueError("`moran_loc` is not a\n " +
                             "esda.moran.Moran_Local instance")
        if 'color' in scatter_kwds or 'c' in scatter_kwds or 'cmap' in scatter_kwds:
            warnings.warn('To change the color use cmap with a colormap of 5,\n' +
                          ' color defines the LISA category')

        # colors
        spots = moran_hot_cold_spots(moran_loc, p)
        hmap = colors.ListedColormap(['#bababa', '#d7191c', '#abd9e9',
                                      '#2c7bb6', '#fdae61'])

    # define customization
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('s', 40)
    fitline_kwds.setdefault('alpha', 0.9)

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7,7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Spatial Lag')
    ax.set_title('Moran Local Scatterplot')

    # plot and set standards
    if zstandard is True:
        lag = lag_spatial(moran_loc.w, moran_loc.z)
        fit = OLS(moran_loc.z[:, None], lag[:, None])
        # v- and hlines
        ax.axvline(0, alpha=0.5, color='k', linestyle='--')
        ax.axhline(0, alpha=0.5, color='k', linestyle='--')
        if p is not None:
            fitline_kwds.setdefault('color', 'k')
            scatter_kwds.setdefault('cmap', hmap)
            scatter_kwds.setdefault('c', spots)
            ax.plot(lag, fit.predy, **fitline_kwds)
            ax.scatter(moran_loc.z, fit.predy,
                       **scatter_kwds)
        else:
            scatter_kwds.setdefault('color', splot_colors['moran_base'])
            fitline_kwds.setdefault('color', splot_colors['moran_fit'])
            ax.plot(lag, fit.predy, **fitline_kwds)
            ax.scatter(moran_loc.z, fit.predy, **scatter_kwds)
    else:
        lag = lag_spatial(moran_loc.w, moran_loc.y)
        b, a = np.polyfit(moran_loc.y, lag, 1)
        # dashed vert at mean of the attribute
        ax.vlines(moran_loc.y.mean(), lag.min(), lag.max(), alpha=0.5,
                  linestyle='--')
        # dashed horizontal at mean of lagged attribute
        ax.hlines(lag.mean(), moran_loc.y.min(), moran_loc.y.max(), alpha=0.5,
                  linestyle='--')
        if p is not None:
            fitline_kwds.setdefault('color', 'k')
            scatter_kwds.setdefault('cmap', hmap)
            scatter_kwds.setdefault('c', spots)
            ax.plot(moran_loc.y, a + b*moran_loc.y, **fitline_kwds)
            ax.scatter(moran_loc.y, lag, **scatter_kwds)
        else:
            scatter_kwds.setdefault('c', splot_colors['moran_base'])
            fitline_kwds.setdefault('color', splot_colors['moran_fit'])
            ax.plot(moran_loc.y, a + b*moran_loc.y, **fitline_kwds)
            ax.scatter(moran_loc.y, lag, **scatter_kwds)
    return fig, ax


def lisa_cluster(moran_loc, gdf, p=0.05, ax=None,
                 legend=True, legend_kwds=None, **kwargs):
    """
    Create a LISA Cluster map

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local or Moran_Local_BV instance
        Values of Moran's Local Autocorrelation Statistic
    gdf : geopandas dataframe instance
        The Dataframe containing information to plot. Note that `gdf` will be
        modified, so calling functions should use a copy of the user
        provided `gdf`. (either using gdf.assign() or gdf.copy())
    p : float, optional
        The p-value threshold for significance. Points will
        be colored by significance.
    ax : matplotlib Axes instance, optional
        Axes in which to plot the figure in multiple Axes layout.
        Default = None
    legend : boolean, optional
        If True, legend for maps will be depicted. Default = True
    legend_kwds : dict, optional
        Dictionary to control legend formatting options. Example:
        ``legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (0.92, 1.05)}``
        Default = None
    **kwargs : keyword arguments, optional
        Keywords designing and passed to geopandas.GeoDataFrame.plot().

    Returns
    -------
    fig : matplotlip Figure instance
        Figure of LISA cluster map
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_Local
    >>> from pysal.viz.splot.esda import lisa_cluster

    Data preparation and statistical analysis
    
    >>> link = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> moran_loc = Moran_Local(y, w)

    Plotting
    
    >>> fig = lisa_cluster(moran_loc, gdf)
    >>> plt.show()
    
    """
    # retrieve colors5 and labels from mask_local_auto
    _, colors5, _, labels = mask_local_auto(moran_loc, p=p)

    # define ListedColormap
    hmap = colors.ListedColormap(colors5)

    if ax is None:
        figsize = kwargs.pop('figsize', None)
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf.assign(cl=labels).plot(column='cl', categorical=True,
                               k=2, cmap=hmap, linewidth=0.1, ax=ax,
                               edgecolor='white', legend=legend,
                               legend_kwds=legend_kwds, **kwargs)
    ax.set_axis_off()
    ax.set_aspect('equal')
    return fig, ax


def plot_local_autocorrelation(moran_loc, gdf, attribute, p=0.05,
                               region_column=None, mask=None,
                               mask_color='#636363', quadrant=None,
                               aspect_equal=True,
                               legend=True, scheme='Quantiles',
                               cmap='YlGnBu', figsize=(15, 4),
                               scatter_kwds=None, fitline_kwds=None):
    '''
    Produce three-plot visualisation of Moran Scatteprlot, LISA cluster
    and Choropleth maps, with Local Moran region and quadrant masking

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local or Moran_Local_BV instance
        Values of Moran's Local Autocorrelation Statistic
    gdf : geopandas dataframe
        The Dataframe containing information to plot the two maps.
    attribute : str
        Column name of attribute which should be depicted in Choropleth map.
    p : float, optional
        The p-value threshold for significance. Points and polygons will
        be colored by significance. Default = 0.05.
    region_column: string, optional
        Column name containing mask region of interest. Default = None
    mask: str, optional
        Identifier or name of the region to highlight. Default = None
    mask_color: str, optional
        Color of mask. Default = '#636363'
    quadrant : int, optional
        Quadrant 1-4 in scatterplot masking values in LISA cluster and
        Choropleth maps. Default = None
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    figsize: tuple, optional
        W, h of figure. Default = (15,4)
    legend: boolean, optional
        If True, legend for maps will be depicted. Default = True
    scheme: str, optional
        Name of PySAL classifier to be used. Default = 'Quantiles'
    cmap: str, optional
        Name of matplotlib colormap used for plotting the Choropleth.
        Default = 'YlGnBu'
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline
        in the scatterplot. Default =None.

    Returns
    -------
    fig : Matplotlib figure instance
        Moran Scatterplot, LISA cluster map and Choropleth.
    axs : list of Matplotlib axes
        Lisat of Matplotlib axes plotted.

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_Local
    >>> from pysal.viz.splot.esda import plot_local_autocorrelation

    Data preparation and analysis
    
    >>> link = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link)
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> moran_loc = Moran_Local(y, w)

    Plotting with quadrant mask and region mask
    
    >>> fig = plot_local_autocorrelation(moran_loc, gdf, 'Donatns', p=0.05,
    ...                                  region_column='Dprtmnt',
    ...                                  mask=['Ain'], quadrant=1)
    >>> plt.show()
    
    '''
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                            subplot_kw={'aspect': 'equal', 'adjustable':'datalim'})
    # Moran Scatterplot
    moran_scatterplot(moran_loc, p=p, ax=axs[0],
                      scatter_kwds=scatter_kwds, fitline_kwds=fitline_kwds)
    if aspect_equal is True:
        axs[0].set_aspect('equal', 'datalim')
    else:
        axs[0].set_aspect('auto')

    # Lisa cluster map
    # TODO: Fix legend_kwds: display boxes instead of points
    lisa_cluster(moran_loc, gdf, p=p, ax=axs[1], legend=legend,
                 legend_kwds={'loc': 'upper left',
                 'bbox_to_anchor': (0.92, 1.05)})
    axs[1].set_aspect('equal')

    # Choropleth for attribute
    gdf.plot(column=attribute, scheme=scheme, cmap=cmap,
             legend=legend, legend_kwds={'loc': 'upper left',
                                         'bbox_to_anchor': (0.92, 1.05)},
             ax=axs[2], alpha=1)
    axs[2].set_axis_off()
    axs[2].set_aspect('equal')

    # MASKING QUADRANT VALUES
    if quadrant is not None:
        # Quadrant masking in Scatterplot
        mask_angles = {1: 0, 2: 90, 3: 180, 4: 270}   # rectangle angles
        # We don't want to change the axis data limits, so use the current ones
        xmin, xmax = axs[0].get_xlim()
        ymin, ymax = axs[0].get_ylim()
        # We are rotating, so we start from 0 degrees and
        # figured out the right dimensions for the rectangles for other angles
        mask_width = {1: abs(xmax),
                      2: abs(ymax),
                      3: abs(xmin),
                      4: abs(ymin)}
        mask_height = {1: abs(ymax),
                       2: abs(xmin),
                       3: abs(ymin),
                       4: abs(xmax)}
        axs[0].add_patch(patches.Rectangle((0, 0), width=mask_width[quadrant],
                                           height=mask_height[quadrant],
                                           angle=mask_angles[quadrant],
                                           color='#E5E5E5', zorder=-1, alpha=0.8))
        # quadrant selection in maps
        non_quadrant = ~(moran_loc.q == quadrant)
        mask_quadrant = gdf[non_quadrant]
        df_quadrant = gdf.iloc[~non_quadrant]
        union2 = df_quadrant.unary_union.boundary

        # LISA Cluster mask and cluster boundary
        with warnings.catch_warnings():  # temorarily surpress geopandas warning
            warnings.filterwarnings('ignore', category=UserWarning)
            mask_quadrant.plot(column=attribute, scheme=scheme, color='white',
                               ax=axs[1], alpha=0.7, zorder=1)
        gpd.GeoSeries([union2]).plot(linewidth=1, ax=axs[1], color='#E5E5E5')

        # CHOROPLETH MASK
        with warnings.catch_warnings():  # temorarily surpress geopandas warning
            warnings.filterwarnings('ignore', category=UserWarning)
            mask_quadrant.plot(column=attribute, scheme=scheme, color='white',
                           ax=axs[2], alpha=0.7, zorder=1)
        gpd.GeoSeries([union2]).plot(linewidth=1, ax=axs[2], color='#E5E5E5')

    # REGION MASKING
    if region_column is not None:
        # masking inside axs[0] or Moran Scatterplot
        ix = gdf[region_column].isin(mask)
        df_mask = gdf[ix]
        x_mask = moran_loc.z[ix]
        y_mask = lag_spatial(moran_loc.w, moran_loc.z)[ix]
        axs[0].plot(x_mask, y_mask, color=mask_color, marker='o',
                    markersize=14, alpha=.8, linestyle="None", zorder=-1)

        # masking inside axs[1] or Lisa cluster map
        union = df_mask.unary_union.boundary
        gpd.GeoSeries([union]).plot(linewidth=2, ax=axs[1], color=mask_color)

        # masking inside axs[2] or Chloropleth
        gpd.GeoSeries([union]).plot(linewidth=2, ax=axs[2], color=mask_color)
    return fig, axs


def _moran_loc_bv_scatterplot(moran_loc_bv, p=None,
                              aspect_equal=True, ax=None,
                              scatter_kwds=None,
                              fitline_kwds=None):
    """
    Moran Bivariate Scatterplot with option of coloring of Local Moran Statistics

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local_BV instance
        Values of Moran's I Local Autocorrelation Statistics
    p : float, optional
        If given, the p-value threshold for significance. Points will
        be colored by significance. By default it will not be colored.
        Default =None.
    aspect_equal : bool, optional
        If True, Axes of Moran Scatterplot will show the same
        aspect or visual proportions.
    ax : Matplotlib Axes instance, optional
        If given, the Moran plot will be created inside this axis.
        Default =None.
    scatter_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points.
        Default =None.
    fitline_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate Moran Local scatterplot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> import geopandas as gpd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> from pysal.explore.esda.moran import Moran_Local_BV
    >>> from pysal.viz.splot.esda import moran_scatterplot
    
    Load data and calculate Moran Local statistics
    
    >>> link = examples.get_path('Guerry.shp')
    >>> gdf = gpd.read_file(link)
    >>> x = gdf['Suicids'].values
    >>> y = gdf['Donatns'].values
    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> m = Moran_Local_BV(x, y, w)
    
    Plot
    
    >>> moran_scatterplot(m)
    >>> plt.show()
    
    Customize plot
    
    >>> moran_scatterplot(m, p=0.05,
    ...                          fitline_kwds=dict(color='#4393c3')))
    >>> plt.show()
    
    """
    # to set default as an empty dictionary that is later filled with defaults
    if scatter_kwds is None:
        scatter_kwds = dict()
    if fitline_kwds is None:
        fitline_kwds = dict()

    if p is not None:
        if not isinstance(moran_loc_bv, Moran_Local_BV):
            raise ValueError("`moran_loc_bv` is not a\n" +
                             "esda.moran.Moran_Local_BV instance")
        if 'color' in scatter_kwds or 'c' in scatter_kwds or 'cmap' in scatter_kwds:
            warnings.warn("To change the color use cmap with a colormap of 5,\n" +
                          "c defines the LISA category, color will interfere with c")

        # colors
        spots_bv = moran_hot_cold_spots(moran_loc_bv, p)
        hmap = colors.ListedColormap(['#bababa', '#d7191c', '#abd9e9',
                                      '#2c7bb6', '#fdae61'])

    # define customization
    scatter_kwds.setdefault('alpha', 0.6)
    scatter_kwds.setdefault('s', 40)
    fitline_kwds.setdefault('alpha', 0.9)

    # get fig and ax
    fig, ax = _create_moran_fig_ax(ax, figsize=(7,7),
                                   aspect_equal=aspect_equal)
    
    # set labels
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Spatial Lag')
    ax.set_title('Moran BV Local Scatterplot')

    # plot and set standards
    lag = lag_spatial(moran_loc_bv.w, moran_loc_bv.zy)
    fit = OLS(moran_loc_bv.zy[:, None], lag[:, None])
    # v- and hlines
    ax.axvline(0, alpha=0.5, color='k', linestyle='--')
    ax.axhline(0, alpha=0.5, color='k', linestyle='--')
    if p is not None:
        fitline_kwds.setdefault('color', 'k')
        scatter_kwds.setdefault('cmap', hmap)
        scatter_kwds.setdefault('c', spots_bv)
        ax.plot(lag, fit.predy, **fitline_kwds)
        ax.scatter(moran_loc_bv.zx, fit.predy,
                   **scatter_kwds)
    else:
        scatter_kwds.setdefault('color', splot_colors['moran_base'])
        fitline_kwds.setdefault('color', splot_colors['moran_fit'])
        ax.plot(lag, fit.predy, **fitline_kwds)
        ax.scatter(moran_loc_bv.zy, fit.predy, **scatter_kwds)
    return fig, ax


def moran_facet(moran_matrix, figsize=(16,12),
                scatter_bv_kwds=None, fitline_bv_kwds=None,
                scatter_glob_kwds=dict(color='#737373'), fitline_glob_kwds=None):
    """
    Moran Facet visualization.
    Includes BV Morans and Global Morans on the diagonal.
    
    Parameters
    ----------
    moran_matrix : esda.moran.Moran_BV_matrix instance
        Dictionary of Moran_BV objects
    figsize : tuple, optional
        W, h of figure. Default =(16,12)
    scatter_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        off-diagonal Moran_BV plots.
        Default =None.
    fitline_bv_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        off-diagonal Moran_BV plots.
        Default =None.
    scatter_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the scatter points of
        diagonal Moran plots.
        Default =None.
    fitline_glob_kwds : keyword arguments, optional
        Keywords used for creating and designing the moran fitline of
        diagonal Moran plots.
        Default =None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Bivariate Moran Local scatterplot figure
    axarr : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    Imports
    
    >>> import matplotlib.pyplot as plt
    >>> import pysal.lib as lp
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from pysal.explore.esda.moran import Moran_BV_matrix
    >>> from pysal.viz.splot.esda import moran_facet
    
    Load data and calculate Moran Local statistics
    
    >>> f = gpd.read_file(lp.examples.get_path("sids2.dbf"))
    >>> varnames = ['SIDR74',  'SIDR79',  'NWR74',  'NWR79']
    >>> vars = [np.array(f[var]) for var in varnames]
    >>> w = lp.io.open(lp.examples.get_path("sids2.gal")).read()
    >>> moran_matrix = Moran_BV_matrix(vars,  w,  varnames = varnames)
    
    Plot
    
    >>> fig, axarr = moran_facet(moran_matrix)
    >>> plt.show()
    
    Customize plot
    
    >>> fig, axarr = moran_facet(moran_matrix,
    ...                          fitline_bv_kwds=dict(color='#4393c3'))
    >>> plt.show()
    
    """
    nrows = int(np.sqrt(len(moran_matrix))) + 1
    ncols = nrows
    
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize,
                              sharey=True, sharex=True)
    fig.suptitle('Moran Facet')
    
    for row in range(nrows):
        for col in range(ncols):
            if row == col:
                global_m = Moran(moran_matrix[row, (row+1) % 4].zy,
                                 moran_matrix[row, (row+1) % 4].w)
                _moran_global_scatterplot(global_m, ax= axarr[row,col],
                                  scatter_kwds=scatter_glob_kwds,
                                  fitline_kwds=fitline_glob_kwds)
                axarr[row, col].set_facecolor('#d9d9d9')
            else:
                _moran_bv_scatterplot(moran_matrix[row,col],
                                  ax=axarr[row,col], 
                                  scatter_kwds=scatter_bv_kwds,
                                  fitline_kwds=fitline_bv_kwds)
    
            axarr[row, col].spines['bottom'].set_visible(False)
            axarr[row, col].spines['left'].set_visible(False)
            if row == nrows - 1:
                axarr[row, col].set_xlabel(str(
                    moran_matrix[(col+1)%4, col].varnames['x']).format(col))
                axarr[row, col].spines['bottom'].set_visible(True)
            else:
                axarr[row, col].set_xlabel('')
                
            if col == 0:
                axarr[row, col].set_ylabel(('Spatial Lag of '+str(
                    moran_matrix[row, (row+1)%4].varnames['y'])).format(row))
                axarr[row, col].spines['left'].set_visible(True)
            else:
                axarr[row, col].set_ylabel('')
                
            axarr[row, col].set_title('') 
    plt.tight_layout()
    return fig, axarr