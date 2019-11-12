import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pysal.explore.esda.moran import Moran_Local
import ipywidgets as widgets
from ipywidgets import interact, fixed
from pysal.explore.giddy.directional import Rose

from ._viz_utils import moran_hot_cold_spots
from ._viz_esda_mpl import lisa_cluster

"""
Lightweight visualizations for pysal dynamics using Matplotlib and Geopandas

TODO
* implement LIMA
* allow for different patterns or list of str
    in dynamic_lisa_composite_explore()
"""

__author__ = ("Stefanie Lumnitz <stefanie.lumitz@gmail.com>")


def _dynamic_lisa_heatmap_data(moran_locy, moran_locx, p=0.05):
    '''
    Utility function to calculate dynamic lisa heatmap table
    and diagonal color mask
    '''
    clustery = moran_hot_cold_spots(moran_locy, p=p)
    clusterx = moran_hot_cold_spots(moran_locx, p=p)

    # to put into seaborn function
    # and set diagonal elements to zero to see the rest better
    heatmap_data = np.zeros((5, 5), dtype=int)
    mask = np.zeros((5, 5), dtype=bool)
    for row in range(5):
        for col in range(5):
            yr1 = clustery == row
            yr2 = clusterx == col
            heatmap_data[row, col] = (yr1 & yr2).sum()
            if row == col:
                mask[row, col] = True
    return heatmap_data, mask


def _moran_loc_from_rose_calc(rose):
    """
    Calculate esda.moran.Moran_Local values from pysal.explore.giddy.rose object
    """
    old_state = np.random.get_state()
    moran_locy = Moran_Local(rose.Y[:, 0], rose.w)
    np.random.set_state(old_state)
    moran_locx = Moran_Local(rose.Y[:, 1], rose.w)
    np.random.set_state(old_state)
    return moran_locy, moran_locx


def dynamic_lisa_heatmap(rose, p=0.05, ax=None, **kwargs):
    """
    Heatmap indicating significant transition of LISA values
    over time inbetween Moran Scatterplot quadrants

    Parameters
    ----------
    rose : giddy.directional.Rose instance
        A ``Rose`` object, which contains (among other attributes) LISA
        values at two points in time, and a method
        to perform inference on those.
    p : float, optional
        The p-value threshold for significance. Default =0.05
    ax : Matplotlib Axes instance, optional
        If given, the figure will be created inside this axis.
        Default =None.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the heatmap.
        These are passed on to `seaborn.heatmap()`.
        See `seaborn` documentation for valid keywords.
        Note: "Start time" refers to `y1` in `Y = np.array([y1, y2]).T`
        with `giddy.Rose(Y, w, k=5)`, "End time" referst to `y2`.

    Returns
    -------
    fig : Matplotlib Figure instance
        Heatmap figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pysal.explore.giddy.directional import Rose
    >>> from pysal.viz.splot.giddy import dynamic_lisa_heatmap

    get csv and shp files

    >>> shp_link = examples.get_path('us48.shp')
    >>> df = gpd.read_file(shp_link)
    >>> income_table = pd.read_csv(examples.get_path("usjoin.csv"))

    calculate relative values

    >>> for year in range(1969, 2010):
    ...     income_table[str(year) + '_rel'] = (
    ...         income_table[str(year)] / income_table[str(year)].mean())
    
    merge to one gdf

    >>> gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')

    retrieve spatial weights and data for two points in time

    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> y1 = gdf['1969_rel'].values
    >>> y2 = gdf['2000_rel'].values

    calculate rose Object

    >>> Y = np.array([y1, y2]).T
    >>> rose = Rose(Y, w, k=5)

    plot

    >>> dynamic_lisa_heatmap(rose)
    >>> plt.show()

    customize plot

    >>> dynamic_lisa_heatmap(rose, cbar='GnBu')
    >>> plt.show()

    """
    moran_locy, moran_locx = _moran_loc_from_rose_calc(rose)
    fig, ax = _dynamic_lisa_heatmap(moran_locy, moran_locx,
                                    p=p, ax=ax, **kwargs)
    return fig, ax


def _dynamic_lisa_heatmap(moran_locy, moran_locx, p, ax, **kwargs):
    """
    Create dynamic_lisa_heatmap figure from pysal.explore.esda.moran.Moran_local values
    """
    heatmap_data, diagonal_mask = _dynamic_lisa_heatmap_data(moran_locy,
                                                             moran_locx, p)
    # set default plot style
    annot = kwargs.pop('annot', True)
    cmap = kwargs.pop('cmap', "YlGnBu")
    mask = kwargs.pop('mask', diagonal_mask)
    cbar = kwargs.pop('cbar', False)
    square = kwargs.pop('square', True)

    # set name for tick labels
    xticklabels = kwargs.pop('xticklabels', ['ns', 'HH', 'HL', 'LH', 'LL'])
    yticklabels = kwargs.pop('yticklabels', ['ns', 'HH', 'HL', 'LH', 'LL'])

    ax = sns.heatmap(heatmap_data, annot=annot, cmap=cmap,
                     xticklabels=xticklabels, yticklabels=yticklabels,
                     mask=mask, ax=ax, cbar=cbar, square=square, **kwargs)
    ax.set_xlabel('End time')
    ax.set_ylabel('Start time')
    fig = ax.get_figure()
    return fig, ax


def dynamic_lisa_rose(rose, attribute=None, ax=None, **kwargs):
    """
    Plot dynamic LISA values in a rose diagram.

    Parameters
    ----------
    rose : giddy.directional.Rose instance
        A ``Rose`` object, which contains (among other attributes) LISA
        values at two points in time, and a method
        to perform inference on those.
    attribute : (n,) ndarray, optional
        Points will be colored by chosen attribute values.
        Variable to specify colors of the colorbars. Default =None.
    ax : Matplotlib Axes instance, optional
        If given, the figure will be created inside this axis.
        Default =None. Note: This axis should have a polar projection.
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the
        `matplotlib.pyplot.scatter()`.
        Note: 'c' and 'color' cannot be passed when attribute is not None.

    Returns
    -------
    fig : Matplotlib Figure instance
        LISA rose plot figure
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pysal.explore.giddy.directional import Rose
    >>> from pysal.viz.splot.giddy import dynamic_lisa_rose

    get csv and shp files

    >>> shp_link = examples.get_path('us48.shp')
    >>> df = gpd.read_file(shp_link)
    >>> income_table = pd.read_csv(examples.get_path("usjoin.csv"))

    calculate relative values

    >>> for year in range(1969, 2010):
    ...     income_table[str(year) + '_rel'] = (
    ...         income_table[str(year)] / income_table[str(year)].mean())
    
    merge to one gdf
    
    >>> gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')

    retrieve spatial weights and data for two points in time

    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> y1 = gdf['1969_rel'].values
    >>> y2 = gdf['2000_rel'].values

    calculate rose Object

    >>> Y = np.array([y1, y2]).T
    >>> rose = Rose(Y, w, k=5)

    plot

    >>> dynamic_lisa_rose(rose, attribute=y1)
    >>> plt.show()

    customize plot

    >>> dynamic_lisa_rose(rose, c='r')
    >>> plt.show()

    """
    # save_old default values
    old_gridcolor = mpl.rcParams['grid.color']
    old_facecolor = mpl.rcParams['axes.facecolor']
    old_edgecolor = mpl.rcParams['axes.edgecolor']
    # define plotting style
    mpl.rcParams['grid.color'] = 'w'
    mpl.rcParams['axes.edgecolor'] = 'w'
    mpl.rcParams['axes.facecolor'] = '#E5E5E5'
    alpha = kwargs.pop('alpha', 0.9)
    cmap = kwargs.pop('cmap', 'YlGnBu')

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        can_insert_colorbar = True
    else:
        fig = ax.get_figure()
        can_insert_colorbar = False

    ax.set_rlabel_position(315)

    if attribute is None:
        c = ax.scatter(rose.theta, rose.r,
                       alpha=alpha, cmap=cmap, **kwargs)
    else:
        if 'c' in kwargs.keys() or 'color' in kwargs.keys():
            raise ValueError('c and color are not valid keywords here; '
                             'attribute is used for coloring')

        c = ax.scatter(rose.theta, rose.r, c=attribute,
                       alpha=alpha, cmap=cmap, **kwargs)
        if can_insert_colorbar:
            fig.colorbar(c)

    # reset style to old default values
    mpl.rcParams['grid.color'] = old_gridcolor
    mpl.rcParams['axes.facecolor'] = old_facecolor
    mpl.rcParams['axes.edgecolor'] = old_edgecolor
    return fig, ax


def _add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.

    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    line.axes.annotate('', xytext=(xdata[0], ydata[0]),
                       xy=(xdata[1], ydata[1]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size)


def dynamic_lisa_vectors(rose, ax=None,
                         arrows=True, **kwargs):
    """
    Plot vectors of positional transition of LISA values
    in Moran scatterplot

    Parameters
    ----------
    rose : giddy.directional.Rose instance
        A ``Rose`` object, which contains (among other attributes) LISA
        values at two points in time, and a method
        to perform inference on those.
    ax : Matplotlib Axes instance, optional
        If given, the figure will be created inside this axis.
        Default =None.
    arrows : boolean, optional
        If True show arrowheads of vectors. Default =True
    **kwargs : keyword arguments, optional
        Keywords used for creating and designing the `matplotlib.pyplot.plot()`
        Note: 'c' and 'color' cannot be passed when attribute is not None.

    Returns
    -------
    fig : Matplotlib Figure instance
        Figure of dynamic LISA vectors
    ax : matplotlib Axes instance
        Axes in which the figure is plotted

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> from pysal.explore.giddy.directional import Rose
    >>> from pysal.viz.splot.giddy import dynamic_lisa_vectors

    get csv and shp files

    >>> shp_link = examples.get_path('us48.shp')
    >>> df = gpd.read_file(shp_link)
    >>> income_table = pd.read_csv(examples.get_path("usjoin.csv"))

    calculate relative values

    >>> for year in range(1969, 2010):
    ...     income_table[str(year) + '_rel'] = (
    ...         income_table[str(year)] / income_table[str(year)].mean())

    merge to one gdf

    >>> gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')

    retrieve spatial weights and data for two points in time

    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> y1 = gdf['1969_rel'].values
    >>> y2 = gdf['2000_rel'].values

    calculate rose Object

    >>> Y = np.array([y1, y2]).T
    >>> rose = Rose(Y, w, k=5)

    plot

    >>> dynamic_lisa_vectors(rose)
    >>> plt.show()

    customize plot

    >>> dynamic_lisa_vectors(rose, arrows=False, c='r')
    >>> plt.show()

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        can_insert_colorbar = True
    else:
        fig = ax.get_figure()
        can_insert_colorbar = False

    xlim = [rose.Y.min(), rose.Y.max()]
    ylim = [rose.wY.min(), rose.wY.max()]

    color = kwargs.pop('color', 'b')
    can_insert_colorbar = False

    xs = []
    ys = []
    for i in range(len(rose.Y)):
        # Plot a vector from xy_start to xy_end
        xs.append(rose.Y[i, :])
        ys.append(rose.wY[i, :])

    xs = np.asarray(xs).T
    ys = np.asarray(ys).T
    lines = ax.plot(xs, ys, color=color, **kwargs)
    if can_insert_colorbar:
        fig.colorbar(lines)

    if arrows:
        for line in lines:
            _add_arrow(line)

    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax


def dynamic_lisa_composite(rose, gdf,
                           p=0.05, figsize=(13, 10)):
    """
    Composite visualisation for dynamic LISA values over two points in time.
    Includes dynamic lisa heatmap, dynamic lisa rose plot,
    and LISA cluster plots for both, compared points in time.

    Parameters
    ----------
    rose : giddy.directional.Rose instance
        A ``Rose`` object, which contains (among other attributes) LISA
        values at two points in time, and a method
        to perform inference on those.
    gdf : geopandas dataframe instance
        The GeoDataFrame containing information and polygons to plot.
    p : float, optional
        The p-value threshold for significance. Default =0.05.
    figsize: tuple, optional
        W, h of figure. Default =(13,10)

    Returns
    -------
    fig : Matplotlib Figure instance
        Dynamic lisa composite figure.
    axs : matplotlib Axes instance
        Axes in which the figure is plotted.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pysal.explore.giddy.directional import Rose
    >>> from pysal.viz.splot.giddy import dynamic_lisa_composite

    get csv and shp files

    >>> shp_link = examples.get_path('us48.shp')
    >>> df = gpd.read_file(shp_link)
    >>> income_table = pd.read_csv(examples.get_path("usjoin.csv"))

    calculate relative values

    >>> for year in range(1969, 2010):
    ...     income_table[str(year) + '_rel'] = (
    ...         income_table[str(year)] / income_table[str(year)].mean())

    merge to one gdf

    >>> gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')

    retrieve spatial weights and data for two points in time

    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> y1 = gdf['1969_rel'].values
    >>> y2 = gdf['2000_rel'].values

    calculate rose Object

    >>> Y = np.array([y1, y2]).T
    >>> rose = Rose(Y, w, k=5)

    plot

    >>> dynamic_lisa_composite(rose, gdf)
    >>> plt.show()

    customize plot

    >>> fig, axs = dynamic_lisa_composite(rose, gdf)
    >>> axs[0].set_ylabel('1996')
    >>> axs[0].set_xlabel('2009')
    >>> axs[1].set_title('LISA cluster for 1996')
    >>> axs[3].set_title('LISA clsuter for 2009')
    >>> plt.show()

    """
    # Moran_Local uses random numbers,
    # which we cannot change between the two years!
    moran_locy, moran_locx = _moran_loc_from_rose_calc(rose)

    # initialize figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Space-time autocorrelation', fontsize=20)
    axs = []
    axs.append(plt.subplot(221))
    axs.append(plt.subplot(222))
    # save_old default values
    old_gridcolor = mpl.rcParams['grid.color']
    old_facecolor = mpl.rcParams['axes.facecolor']
    old_edgecolor = mpl.rcParams['axes.edgecolor']
    # define plotting style
    mpl.rcParams['grid.color'] = 'w'
    mpl.rcParams['axes.edgecolor'] = 'w'
    mpl.rcParams['axes.facecolor'] = '#E5E5E5'
    # define axs[2]
    axs.append(plt.subplot(223, projection='polar'))
    # reset style to old default values
    mpl.rcParams['grid.color'] = old_gridcolor
    mpl.rcParams['axes.facecolor'] = old_facecolor
    mpl.rcParams['axes.edgecolor'] = old_edgecolor
    # define axs[3]
    axs.append(plt.subplot(224))

    # space_time_heatmap
    _dynamic_lisa_heatmap(moran_locy, moran_locx, p=p, ax=axs[0])
    axs[0].xaxis.set_ticks_position('top')
    axs[0].xaxis.set_label_position('top')

    # Lisa_cluster maps
    lisa_cluster(moran_locy, gdf, p=p, ax=axs[1], legend=True,
                 legend_kwds={'loc': 'upper left',
                 'bbox_to_anchor': (0.92, 1.05)})
    axs[1].set_title('Start time')
    lisa_cluster(moran_locx, gdf, p=p, ax=axs[3], legend=True,
                 legend_kwds={'loc': 'upper left',
                 'bbox_to_anchor': (0.92, 1.05)})
    axs[3].set_title('End time')

    # Rose diagram: Moran movement vectors:
    dynamic_lisa_rose(rose, ax=axs[2])
    return fig, axs


def _dynamic_lisa_widget_update(rose, gdf, start_time, end_time,
                                p=0.05, figsize=(13, 10)):
    """
    Update rose values if widgets are used
    """
    # determine rose object for (timex, timey),
    # which comes from interact widgets
    y1 = gdf[start_time].values
    y2 = gdf[end_time].values
    Y = np.array([y1, y2]).T
    rose_update = Rose(Y, rose.w, k=5)

    fig, _ = dynamic_lisa_composite(rose_update, gdf, p=p, figsize=figsize)


def dynamic_lisa_composite_explore(rose, gdf, pattern='',
                                   p=0.05, figsize=(13, 10)):
    """
    Interactive exploration of dynamic LISA values
    for different dates in a dataframe.
    Note: only possible in jupyter notebooks

    Parameters
    ----------
    rose : giddy.directional.Rose instance
        A ``Rose`` object, which contains (among other attributes)
        weights to calculate `esda.moran.Moran_local` values
    gdf : geopandas dataframe instance
        The Dataframe containing information and polygons to plot.
    pattern : str, optional
        Option to extract all columns ending with a specific pattern.
        Only extracted columns will be used for comparison.
    p : float, optional
        The p-value threshold for significance. Default =0.05
    figsize: tuple, optional
        W, h of figure. Default =(13,10)

    Returns
    -------
    None

    Examples
    --------
    **Note**: this function creates Jupyter notebook widgets, so is meant only
    to run in a notebook.

    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from pysal.lib.weights.contiguity import Queen
    >>> from pysal.lib import examples
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    If you want to see figures embedded inline in a Jupyter notebook,
    add a line ``%matplotlib inline`` at the top of your notebook.

    >>> from pysal.explore.giddy.directional import Rose
    >>> from pysal.viz.splot.giddy import dynamic_lisa_composite_explore

    get csv and shp files

    >>> shp_link = examples.get_path('us48.shp')
    >>> df = gpd.read_file(shp_link)
    >>> income_table = pd.read_csv(examples.get_path("usjoin.csv"))

    calculate relative values

    >>> for year in range(1969, 2010):
    ...     income_table[str(year) + '_rel'] = (
    ...         income_table[str(year)] / income_table[str(year)].mean())

    merge to one gdf

    >>> gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')

    retrieve spatial weights and data for two points in time

    >>> w = Queen.from_dataframe(gdf)
    >>> w.transform = 'r'
    >>> y1 = gdf['1969_rel'].values
    >>> y2 = gdf['2000_rel'].values

    calculate rose Object

    >>> Y = np.array([y1, y2]).T
    >>> rose = Rose(Y, w, k=5)

    plot

    >>> fig = dynamic_lisa_composite_explore(rose, gdf, pattern='rel')
    >>> # plt.show()

    """
    coldict = {col: col for col in gdf.columns if
               col.endswith(pattern)}
    interact(_dynamic_lisa_widget_update,
             start_time=coldict, end_time=coldict, rose=fixed(rose),
             gdf=fixed(gdf), p=fixed(p), figsize=fixed(figsize))
