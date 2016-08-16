
get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mpc

try:
    import brewer2mpl
    from brewer2mpl import qualitative as b2m_qualitative
    from brewer2mpl import sequential as b2m_sequential
    from brewer2mpl import diverging as b2m_diverging
except:
    print('brewer2mpl  not installed. Functionality '
          'related to it will not work')

from ipywidgets import Dropdown, RadioButtons,  HBox, interact
from IPython.display import display
from pysal.contrib.viz import mapping as maps
from pysal.contrib.viz.color import get_maps_by_type


def get_color_map_b2m(name='BuGn', cmtype='sequential', k=5,
                      color_encoding='hexc'):

    """
    DEPRECATED!!!

    Get a brewer colormap from `brewer2mpl`

    Arguments
    ---------

    name:   string
            colormap name

    cmtype: string
            colormap scale type  [sequential, diverging, qualitative]

    k:  int
        number of classes

    color_encoding: string
                    encoding of colors [hexc, rgb, mpl, mpl_colormap]
                    hex: list of hex strings
                    rgb: list of RGB 0-255 triplets
                    mpl: list of RGB 0-1 triplets as used by matplotlib
                    mpl_colormap: matplotlib color map

    Returns

    colors:  color map in the specified color_encoding

    """
    encs = {'hexc': 'hex_colors',
            'rgb': 'colors',
            'mpl': 'mpl_colors',
            'mpl_colormap': 'mpl_colormap'}
    try:
        bmap = brewer2mpl.get_map(name, cmtype, k)
        colors = getattr(bmap, encs[color_encoding.lower()])
        return colors
    except:
        print('Color map not found: ', name, cmtype, k)

# get maps for each ctype and default k=5 for populating display options
ctypes = (b2m_sequential, b2m_diverging, b2m_qualitative)
color_display_types = {}
for ctype in ctypes:
    cmaps = get_maps_by_type(ctype)
    ctype_name = ctype.__name__.split(".")[1]
    displays = {}
    for cmap in cmaps:
        c = get_color_map_b2m(cmtype=ctype_name, name=cmap)
        displays[cmap] = c
    color_display_types[ctype_name] = displays


def pcmaps(dtype, selected=0):
    """
    Embed a figure displaying color maps for a given data type and mimic a
    selector
    """

    fig = plt.figure(figsize=(2, 2))
    w = 1. / (9 + .5 * (9-1))
    h = 1./11
    ax = fig.add_subplot(111, aspect='equal')
    for i, cmap in enumerate(color_display_types[dtype]):
        c = [mpc.hex2color(c) for c in color_display_types[dtype][cmap]]
        row = i / 9
        col = i % 9
        lx = col * (w + .5 * w)
        ly = (1-row) * (5 * h + h)
        for j, clr in enumerate(c):
            p = patches.Rectangle((lx, ly+(j*h)), w, h, fill=True, color=clr)
            ax.add_patch(p)
    # selected
    row = selected / 9
    col = selected % 9
    lx = col * (w + .5*w)
    ly = (1-row) * (5*h+h)
    p = patches.Rectangle(
        (lx, ly), w, 5*h,
        fill=False
        )
    ax.add_patch(p)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    fig.savefig('selected.png', dpi=90, bbox_inches='tight')


def choro(gd, col, show_cmaps=True, fig_size=(6, 6)):
    dt = sorted(color_display_types.keys(), reverse=True)
    data_type = RadioButtons(description='Data Type', options=dt)
    bindings = {}
    for t in dt:
        bindings[t] = color_display_types[t].keys()

    cmap_dd = Dropdown(description='CMap:', options=bindings[data_type.value])

    def type_change(change):
        with cmap_dd.hold_trait_notifications():
            cmap_dd.options = bindings[change['new']]
            k_dd.options = kbindings[change['new']]

    def cmap_change(change):
            with cmap_dd.hold_trait_notifications():
                print('new cmap', str(change['new']))

    data_type.observe(type_change, names=['value'])
    cmap_dd.observe(cmap_change, names=['value'])
    kbindings = {'sequential': map(str, range(3, 9+1)),
                 'qualitative': map(str, range(3, 12+1)),
                 'diverging': map(str, range(3, 11+1))}

    k_dd = Dropdown(description='k', options=kbindings[data_type.value])
    display(HBox([data_type, k_dd]))

    @interact(cmap=cmap_dd)
    def plot_cmaps(cmap):
        i = cmap_dd.options.index(cmap)
        if show_cmaps:
            pcmaps(data_type.value, i)
        maps.geoplot(gd, col, k=int(k_dd.value), palette=str(cmap_dd.value),
                     figsize=fig_size)
