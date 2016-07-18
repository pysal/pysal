"""
color handling for mapping and geovisualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mpc


try:
    import brewer2mpl
    from brewer2mpl import qualitative, sequential, diverging
except:
    print('brewer2mpl  not installed. Functionality '
          'related to it will not work')


def get_color_map(name='BuGn', cmtype='sequential', k=5,
                  color_encoding='hexc'):

    """
    Get a brewer colormap

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


def get_maps_by_type(data_type):
    names = [name for name in dir(data_type) if not name.startswith('_')]
    return names

# get maps for each ctype and default k=5 for populating display options
ctypes = (sequential, diverging, qualitative)
color_display_types = {}
for ctype in ctypes:
    cmaps = get_maps_by_type(ctype)
    ctype_name = ctype.__name__.split(".")[1]
    displays = {}
    for cmap in cmaps:
        c = get_color_map(cmtype=ctype_name, name=cmap)
        displays[cmap] = c
    color_display_types[ctype_name] = displays


def plot_cmaps(dtype, selected=0):
    """
    Embed a figure displaying color maps for a given data type and mimic a
    selector
    """

    fig = plt.figure(figsize=(2,2))
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
