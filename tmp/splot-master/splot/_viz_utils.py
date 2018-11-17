import numpy as np
import mapclassify as classify
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
Utility functions for lightweight visualizations in splot
"""

__author__ = ("Stefanie Lumnitz <stefanie.lumitz@gmail.com>")


def moran_hot_cold_spots(moran_loc, p=0.05):
    sig = 1 * (moran_loc.p_sim < p)
    HH = 1 * (sig * moran_loc.q == 1)
    LL = 3 * (sig * moran_loc.q == 3)
    LH = 2 * (sig * moran_loc.q == 2)
    HL = 4 * (sig * moran_loc.q == 4)
    cluster = HH + LL + LH + HL
    return cluster


def mask_local_auto(moran_loc, p=0.5):
    '''
    Create Mask for coloration and labeling of local spatial autocorrelation

    Parameters
    ----------
    moran_loc : esda.moran.Moran_Local instance
        values of Moran's I Global Autocorrelation Statistic
    p : float
        The p-value threshold for significance. Points will
        be colored by significance.

    Returns
    -------
    cluster_labels : list of str
        List of labels - ['ns', 'HH', 'LH', 'LL', 'HL']
    colors5 : list of str
        List of colours - ['lightgrey', 'red', 'lightblue','blue', 'pink']
    colors : array of str
        Array containing coloration for each input value/ shape.
    labels : list of str
        List of label for each attribute value/ polygon.
    '''
    # create a mask for local spatial autocorrelation
    cluster = moran_hot_cold_spots(moran_loc, p)

    cluster_labels = ['ns', 'HH', 'LH', 'LL', 'HL']
    labels = [cluster_labels[i] for i in cluster]

    colors5 = {0: 'lightgrey',
               1: '#d7191c',
               2: '#abd9e9',
               3: '#2c7bb6',
               4: '#fdae61'}
    colors = [colors5[i] for i in cluster]  # for Bokeh
    # for MPL:
    colors5 = (['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6', 'lightgrey'])

    # HACK need this, because MPL sorts these labels while Bokeh does not
    cluster_labels.sort()
    return cluster_labels, colors5, colors, labels


_classifiers = {
    'box_plot': classify.Box_Plot,
    'equal_interval': classify.Equal_Interval,
    'fisher_jenks': classify.Fisher_Jenks,
    'headtail_breaks': classify.HeadTail_Breaks,
    'jenks_caspall': classify.Jenks_Caspall,
    'jenks_caspall_forced': classify.Jenks_Caspall_Forced,
    'max_p_classifier': classify.Max_P_Classifier,
    'maximum_breaks': classify.Maximum_Breaks,
    'natural_breaks': classify.Natural_Breaks,
    'quantiles': classify.Quantiles,
    'percentiles': classify.Percentiles,
    'std_mean': classify.Std_Mean,
    'user_defined': classify.User_Defined,
    }


def bin_values_choropleth(attribute_values, method='quantiles',
                          k=5):
    '''
    Create bins based on different classification methods.
    Needed for legend labels and Choropleth coloring.

    Parameters
    ----------
    attribute_values : array or geopandas.series instance
        Array containing relevant attribute values.
    method : str
        Classification method to be used. Options supported:
        * 'quantiles' (default)
        * 'fisher-jenks'
        * 'equal-interval'
    k : int
        Number of bins, assigning values to. Default k=5

    Returns
    -------
    bin_values : mapclassify instance
        Object containing bin ids for each observation (.yb),
        upper bounds of each class (.bins), number of classes (.k)
        and number of onservations falling in each class (.counts)
    '''
    if method not in ['quantiles', 'fisher_jenks', 'equal_interval']:
        raise ValueError("Method {} not supported".format(method))

    bin_values = _classifiers[method](attribute_values, k)
    return bin_values


def bin_labels_choropleth(gdf, attribute_values, method='quantiles', k=5):
    '''
    Create labels for each bin in the legend

    Parameters
    ----------
    gdf : Geopandas dataframe
        Dataframe containign relevant shapes and attribute values.
    attribute_values : array or geopandas.series instance
        Array containing relevant attribute values.
    method : str, optional
        Classification method to be used. Options supported:
        * 'quantiles' (default)
        * 'fisher-jenks'
        * 'equal-interval'
    k : int, optional
        Number of bins, assigning values to. Default k=5

    Returns
    -------
    bin_labels : list of str
        List of label for each bin.
    '''
    # Retrieve bin values from bin_values_choropleth()
    bin_values = bin_values_choropleth(attribute_values, method=method, k=k)

    # Extract bin ids (.yb) and upper bounds for each class (.bins)
    yb = bin_values.yb
    bins = bin_values.bins

    # Create bin labels (smaller version)
    bin_edges = bins.tolist()
    bin_labels = []
    for i in range(k):
        bin_labels.append('<{:1.1f}'.format(bin_edges[i]))

    # Add labels (which are the labels printed in the legend) to each row of gdf
    labels = np.array([bin_labels[c] for c in yb])
    gdf['labels_choro'] = [str(l) for l in labels]
    return bin_labels


def add_legend(fig, labels, colors):
    """
    Add a legend to a figure given legend labels & colors.

    Parameters
    ----------
    fig : Bokeh Figure instance
        Figure instance labels should be generated for.
    labels : list of str
        Labels to use as legend entries.
    colors : Bokeh Palette instance
        Palette instance containing colours of choice.
    """
    from bokeh.models import Legend
    # add labels to figure (workaround,
    # legend with geojsondatasource doesn't work,
    # see https://github.com/bokeh/bokeh/issues/5904)
    items = []
    for label, color in zip(labels, colors):
        patch = fig.patches(xs=[], ys=[], fill_color=color)
        items.append((label, [patch]))

    legend = Legend(items=items, location='top_left', margin=0,
                    orientation='horizontal')
    # possibility to define glyph_width=10, glyph_height=10)
    legend.label_text_font_size = '8pt'
    fig.add_layout(legend, 'below')
    return legend


def format_legend(values):
    """
    Helper to return sensible legend values
    
    Parameters
    ----------
    values: array
        Values plotted in legend.
    """
    in_thousand = False
    if np.any(values > 1000):
        in_thousand = True
        values = values / 1000
    return values, in_thousand


def calc_data_aspect(plot_height, plot_width, bounds):
    # Deal with data ranges in Bokeh:
    # make a meter in x and a meter in y the same in pixel lengths
    aspect_box = plot_height / plot_width   # 2 / 1 = 2
    xmin, ymin, xmax, ymax = bounds
    x_range = xmax - xmin  # 1 = 1 - 0
    y_range = ymax - ymin  # 3 = 3 - 0
    aspect_data = y_range / x_range  # 3 / 1 = 3
    if aspect_data > aspect_box:
        # we need to increase x_range,
        # such that aspect_data becomes equal to aspect_box
        halfrange = 0.5 * x_range * (aspect_data / aspect_box - 1)
        # 0.5 * 1 * (3 / 2 - 1) = 0.25
        xmin -= halfrange  # 0 - 0.25 = -0.25
        xmax += halfrange  # 1 + 0.25 = 1.25
    else:
        # we need to increase y_range
        halfrange = 0.5 * y_range * (aspect_box / aspect_data - 1)
        ymin -= halfrange
        ymax += halfrange

    # Add a bit of margin to both x and y
    margin = 0.03
    xmin -= (xmax - xmin) / 2 * margin
    xmax += (xmax - xmin) / 2 * margin
    ymin -= (ymax - ymin) / 2 * margin
    ymax += (ymax - ymin) / 2 * margin
    return xmin, xmax, ymin, ymax


# Utility functions for colormaps
# Color design
splot_colors = dict(moran_base='#bababa',
                    moran_fit='#d6604d')

# Utility function #1 - forces continuous diverging colormap to be centered at zero
def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Parameters
    ----------
    cmap : matplotlib colormap
        colormap to be altered
    start : float, optional
        Offset from lowest point in the colormap's range.
        Should be between 0.0 and `midpoint`.
        Default =0.0 (no lower ofset).
    midpoint : float, optional
        The new center of the colormap.Should be between 0.0 and
        1.0. In general, this should be 1 - vmax/(vmax + abs(vmin)).
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75.
        Default =0.5 (no shift).
    stop : float, optional
        Offset from highets point in the colormap's range.
        Should be between `midpoint` and 1.0.
        Default =1.0 (no upper ofset).
    name : str, optional
        Name of the new colormap.
    
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap) 

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    new_cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)
    return new_cmap


# Utility #2 - truncate colorcap in order to grab only positive or negative portion
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Function to truncate a colormap by selecting a subset of
    the original colormap's values

    Parameters
    ----------
    cmap : Mmatplotlib colormap
        Colormap to be altered
    minval : float, optional
        Minimum value of the original colormap to include
        in the truncated colormap. Default =0.0.
    maxval : Maximum value of the original colormap to
        include in the truncated colormap. Default =1.0.
    n : int, optional
        Number of intervals between the min and max values
        for the gradient of the truncated colormap. Default =100.
          
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
