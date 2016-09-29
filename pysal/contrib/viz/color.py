"""
color handling for mapping and geovisualization
"""

from warnings import warn
try:
    import palettable as pltt
    from palettable.colorbrewer import qualitative, sequential, diverging
except:
    warn('palettable  not installed. Functionality '
          'related to it will not work')

def get_color_map(palette=None, name='BuGn', cmtype='sequential', k=5,
                  color_encoding='hex'):
    """
    Get a brewer colormap from `palettable`

    Arguments
    ---------

    palette         : `palettable` palette
                      Palettable object for a given palette
    name            : string
                      colormap name

    cmtype          : string
                      colormap scale type  [sequential, diverging, qualitative]

    k               : int
                      number of classes

    color_encoding  : string
                      encoding of colors [hexc, rgb, mpl, mpl_colormap]
                      * hex: list of hex strings
                      * rgb: list of RGB 0-255 triplets
                      * mpl: list of RGB 0-1 triplets as used by matplotlib
                      * mpl_colormap: matplotlib color map

    Returns
    -------

    colors:  color map in the specified color_encoding

    """
    encs = {'hex': 'hex_colors',
            'rgb': 'colors',
            'mpl': 'mpl_colors',
            'mpl_colormap': 'mpl_colormap'}
    if not palette:
        cmtype = pltt2type[name.lower()]
        if name[-2:] == '_r':
            palette = pltt.colorbrewer.get_map(name[:-2], cmtype,
                    k, reverse=True)
        else:
            palette = pltt.colorbrewer.get_map(name, cmtype, k)
    colors = getattr(palette, encs[color_encoding.lower()])
    return colors

def _build_pltt2type():
    types = ['sequential', 'diverging', 'qualitative']
    pltt2type = {}
    for t in types:
        pals = list(set([
            p.split('_')[0] for p in dir(getattr(pltt.colorbrewer, t))
                            if p[0]!='_'
                            ]))
        pals = pals + [p+'_r' for p in pals]
        for p in pals:
            pltt2type[p.lower()] = t
    return pltt2type

pltt2type = _build_pltt2type()

def get_maps_by_type(data_type):
    names = [name for name in dir(data_type) if not name.startswith('_')]
    return names

