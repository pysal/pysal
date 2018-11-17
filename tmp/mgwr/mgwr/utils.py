import numpy as np
from libpysal.common import requires

@requires('matplotlib')
def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Parameters
    ----------
    cmap : The matplotlib colormap to be altered
    start : Offset from lowest point in the colormap's range.
      Defaults to 0.0 (no lower ofset). Should be between
      0.0 and `midpoint`.
    midpoint : The new center of the colormap. Defaults to
      0.5 (no shift). Should be between 0.0 and 1.0. In
      general, this should be  1 - vmax/(vmax + abs(vmin))
      For example if your data range from -15.0 to +5.0 and
      you want the center of the colormap at 0.0, `midpoint`
      should be set to  1 - 5/(5 + 15)) or 0.75
    stop : Offset from highets point in the colormap's range.
      Defaults to 1.0 (no upper ofset). Should be between
      `midpoint` and 1.0.
    
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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

@requires('matplotlib')
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Function to truncate a colormap by selecting a subset of the original colormap's values

    Parameters
    ----------
    cmap : Mmatplotlib colormap to be altered
    minval : Minimum value of the original colormap to include in the truncated colormap
    maxval : Maximum value of the original colormap to include in the truncated colormap
    n : Number of intervals between the min and max values for the gradient of the truncated colormap
          
    Returns
    -------
    new_cmap : A new colormap that has been shifted. 
    '''
    
    import matplotlib as mpl

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

@requires('matplotlib')
@requires('geopandas')
def compare_surfaces(data, var1, var2, gwr_t, gwr_bw, mgwr_t, mgwr_bw, name,
        kwargs1, kwargs2, savefig=None):
    '''
    Function that creates comparative visualization of GWR and MGWR surfaces.

    Parameters
    ----------
    data   : pandas or geopandas Dataframe
             gwr/mgwr results
    var1   : string
             name of gwr parameter estimate column in frame
    var2   : string
             name of mgwr parameter estimate column in frame
    gwr_t  : string
             name of gwr t-values column in frame associated with var1
    gwr_bw : float
             bandwidth for gwr model for var1
    mgwr_t : string
             name of mgwr t-values column in frame associated with var2
    mgwr_bw: float
             bandwidth for mgwr model for var2
    name   : string
             common variable name to use for title
    kwargs1:
             additional plotting arguments for gwr surface
    kwargs2:
             additional plotting arguments for mgwr surface
    savefig: string, optional
             path to save the figure. Default is None. Not to save figure.
    '''
    import matplotlib.pyplot as plt
    import geopandas as gp

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
    ax0 = axes[0]
    ax0.set_title('GWR ' + name + ' Surface (BW: ' + str(gwr_bw) +')', fontsize=40)
    ax1 = axes[1]
    ax1.set_title('MGWR ' + name + ' Surface (BW: ' + str(mgwr_bw) +')', fontsize=40)

    #Set color map
    cmap = plt.cm.seismic

    #Find min and max values of the two combined datasets
    gwr_min = data[var1].min()
    gwr_max = data[var1].max()
    mgwr_min = data[var2].min()
    mgwr_max = data[var2].max()
    vmin = np.min([gwr_min, mgwr_min])
    vmax = np.max([gwr_max, mgwr_max])

    #If all values are negative use the negative half of the colormap
    if (vmin < 0) & (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    #If all values are positive use the positive half of the colormap
    elif (vmin > 0) & (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    #Otherwise, there are positive and negative values so the colormap so zero is the midpoint
    else:
        cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)

    #Create scalar mappable for colorbar and stretch colormap across range of data values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    #Plot GWR parameters
    data.plot(var1, cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax, **kwargs1)
    if (gwr_t == 0).any():
        data[gwr_t == 0].plot(color='lightgrey', ax=ax0, **kwargs2)

    #Plot MGWR parameters
    data.plot(var2, cmap=sm.cmap, ax=ax1, vmin=vmin, vmax=vmax, **kwargs1)
    if (mgwr_t == 0).any():
        data[mgwr_t == 0].plot(color='lightgrey', ax=ax1, **kwargs2)

    #Set figure options and plot 
    fig.tight_layout()    
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=50) 
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if savefig is not None:
    	plt.savefig(savefig)
    plt.show()
