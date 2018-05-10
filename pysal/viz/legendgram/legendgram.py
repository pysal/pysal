from .util import make_location as _make_location
import numpy as np

def legendgram(f, ax, y, breaks, pal, bins=50, clip=None,
               loc = 'lower left', legend_size=(.27,.2),
               frameon=False, tick_params = None):
    '''
    Add a histogram in a choropleth with colors aligned with map
    ...
    
    Arguments
    ---------
    f           : Figure
    ax          : AxesSubplot
    y           : ndarray/Series
                  Values to map
    breaks      : list
                  Sequence with breaks for each class (i.e. boundary values
                  for colors)
    pal         : palettable colormap
    rescale     : None/tuple
                  [Optional. Default=None] If a tuple, clips the X
                  axis of the histogram to the bounds provided.
    loc         :   string or int
                    valid legend location like that used in matplotlib.pyplot.legend
    legend_size :
    frameon     :
    tick_params :
    '''
    k = len(breaks)
    assert k == pal.number, "provided number of classes does not match number of colors in palette."
    histpos = _make_location(ax, loc, legend_size=legend_size)

    histax = f.add_axes(histpos)
    N, bins, patches = histax.hist(y, bins=bins, color='0.1')
    #---
    pl = pal.get_mpl_colormap()
    bucket_breaks = [0]+[np.searchsorted(bins, i) for i in breaks]
    for c in range(k):
        for b in range(bucket_breaks[c], bucket_breaks[c+1]):
            patches[b].set_facecolor(pl(c/k))
    #---
    if clip is not None:
        histax.set_xlim(*clip)
    histax.set_frame_on(frameon)
    histax.get_yaxis().set_visible(False)
    if tick_params is None:
        tick_params = dict()
    tick_params['labelsize'] = tick_params.get('labelsize', 12)
    histax.tick_params(**tick_params)
    return histax