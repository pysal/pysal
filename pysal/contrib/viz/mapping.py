"""
Choropleth mapping using PySAL and Matplotlib

ToDo:
    * map_line_shp, map_point_shp should take a shp object not a shp_link
    * Same for map_poly_shp(_lonlat)

"""

__author__ = "Sergio Rey <sjsrey@gmail.com>", "Dani Arribas-Bel <daniel.arribas.bel@gmail.com"


import pandas as pd
import pysal as ps
import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import colors as clrs
import matplotlib as mpl
from matplotlib.pyplot import fill, text
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, PathCollection, PatchCollection

# Low-level pieces

def map_point_shp(shp, which='all', bbox=None):
    '''
    Create a map object from a point shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL point iterable (e.g.
                      shape object from `ps.open` a point shapefile) If it does
                      not contain the attribute `bbox`, it must be passed
                      separately in `bbox`.
    which           : str/list
                      List of booleans for which polygons of the shapefile to
                      be included (True) or excluded (False)
    bbox            : None/list
                      [Optional. Default=None] List with bounding box as in a
                      PySAL object. If nothing is passed, it tries to obtain
                      it as an attribute from `shp`.

    Returns
    -------

    map             : PatchCollection
                      Map object with the points from the shape

    '''
    if not bbox:
        bbox = shp.bbox
    pts = []
    if which == 'all':
        for pt in shp:
                pts.append(pt)
    else:
        for inwhich, pt in zip(which, shp):
            if inwhich:
                    pts.append(pt)
    pts = np.array(pts)
    sc = plt.scatter(pts[:, 0], pts[:, 1])
    #print(sc.get_axes().get_xlim())
    #_ = _add_axes2col(sc, bbox)
    #print(sc.get_axes().get_xlim())
    return sc

def map_line_shp(shp, which='all', bbox=None):
    '''
    Create a map object from a line shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL line iterable (e.g.
                      shape object from `ps.open` a line shapefile) If it does
                      not contain the attribute `bbox`, it must be passed
                      separately in `bbox`.
    which           : str/list
                      List of booleans for which polygons of the shapefile to
                      be included (True) or excluded (False)
    bbox            : None/list
                      [Optional. Default=None] List with bounding box as in a
                      PySAL object. If nothing is passed, it tries to obtain
                      it as an attribute from `shp`.

    Returns
    -------

    map             : PatchCollection
                      Map object with the lines from the shape
                      This includes the attribute `shp2dbf_row` with the
                      cardinality of every line to its row in the dbf
                      (zero-offset)

    '''
    if not bbox:
        bbox = shp.bbox
    patches = []
    rows = []
    i = 0
    if which == 'all':
        for shape in shp:
            for xy in shape.parts:
                patches.append(xy)
                rows.append(i)
            i += 1
    else:
        for inwhich, shape in zip(which, shp):
            if inwhich:
                for xy in shape.parts:
                    patches.append(xy)
                    rows.append(i)
                i += 1
    lc = LineCollection(patches)
    #_ = _add_axes2col(lc, bbox)
    lc.shp2dbf_row = rows
    return lc

def map_poly_shp(shp, which='all', bbox=None):
    '''
    Create a map object from a polygon shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL polygon iterable (e.g.
                      shape object from `ps.open` a poly shapefile) If it does
                      not contain the attribute `bbox`, it must be passed
                      separately in `bbox`.
    which           : str/list
                      List of booleans for which polygons of the shapefile to
                      be included (True) or excluded (False)
    bbox            : None/list
                      [Optional. Default=None] List with bounding box as in a
                      PySAL object. If nothing is passed, it tries to obtain
                      it as an attribute from `shp`.

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shape
                      This includes the attribute `shp2dbf_row` with the
                      cardinality of every polygon to its row in the dbf
                      (zero-offset)

    '''
    if not bbox:
        bbox = shp.bbox
    patches = []
    rows = []
    i = 0
    if which == 'all':
        for shape in shp:
            for ring in shape.parts:
                xy = np.array(ring)
                patches.append(xy)
                rows.append(i)
            i += 1
    else:
        for inwhich, shape in zip(which, shp):
            if inwhich:
                for ring in shape.parts:
                    xy = np.array(ring)
                    patches.append(xy)
                    rows.append(i)
                i += 1
    pc = PolyCollection(patches)
    #_ = _add_axes2col(pc, bbox)
    pc.shp2dbf_row = rows
    return pc

# Mid-level pieces

def setup_ax(polyCos_list, bboxs, ax=None):
    '''
    Generate an Axes object for a list of collections
    ...

    Arguments
    ---------
    polyCos_list: list
                  List of Matplotlib collections (e.g. an object from
                  map_poly_shp)
    bboxs       : list
                  List of lists, each containing the bounding box of the
                  respective polyCo, expressed as [xmin, ymin, xmax, ymax]
    ax          : AxesSubplot
                  (Optional) Pre-existing axes to which append the collections
                  and setup

    Returns
    -------
    ax          : AxesSubplot
                  Rescaled axes object with the collection and without frame
                  or X/Yaxis
    '''
    if not ax:
        ax = plt.axes()

    for polyCo, bbox in zip(polyCos_list, bboxs):
        ax.add_collection(polyCo)
        polyCo.axes.set_xlim((bbox[0], bbox[2]))
        polyCo.axes.set_ylim((bbox[1], bbox[3]))
    abboxs = np.array(bboxs)
    ax.set_xlim((abboxs[:, 0].min(), \
                 abboxs[:, 2].max()))
    ax.set_ylim((abboxs[:, 1].min(), \
                 abboxs[:, 3].max()))
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    return ax

def _add_axes2col(col, bbox):
    """
    Adds (inplace) axes with proper limits to a poly/line collection. This is
    still pretty much a hack! Ideally, you don't have to setup a new figure
    for this
    ...

    Arguments
    ---------
    col     : Collection
    bbox    : list
              Bounding box as [xmin, ymin, xmax, ymax]
    """
    tf = plt.figure()
    ax = plt.axes()
    minx, miny, maxx, maxy = bbox
    ax.set_xlim((minx, maxx))
    ax.set_ylim((miny, maxy))
    col.set_axes(ax)
    plt.close(tf)
    return None

def base_choropleth_classless(map_obj, values, cmap='Greys' ):
    '''
    Set classless coloring from a map object
    ...

    Arguments
    ---------

    map_obj         : Poly/Line collection
                      Output from map_X_shp
    values          : array
                      Numpy array with values to map
    cmap            : str
                      Matplotlib coloring scheme

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      classless coloring

    '''
    cmap = cm.get_cmap(cmap)
    map_obj.set_cmap(cmap)
    if isinstance(map_obj, mpl.collections.PolyCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_array(pvalues)
        map_obj.set_edgecolor('k')
    elif isinstance(map_obj, mpl.collections.LineCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_array(pvalues)
    elif isinstance(map_obj, mpl.collections.PathCollection):
        if not hasattr(map_obj, 'shp2dbf_row'):
            map_obj.shp2dbf_row = np.arange(values.shape[0])
        map_obj.set_array(values)
    return map_obj

def base_choropleth_unique(map_obj, values,  cmap='hot_r'):
    '''
    Set coloring based on unique values from a map object
    ...

    Arguments
    ---------

    map_obj         : Poly/Line collection
                      Output from map_X_shp
    values          : array
                      Numpy array with values to map
    cmap            : dict/str
                      [Optional. Default='hot_r'] Dictionary mapping {value:
                      color}. Alternatively, a string can be passed specifying
                      the Matplotlib coloring scheme for a random assignment
                      of {value: color}

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    '''
    if type(cmap) == str:
        uvals = np.unique(values)
        colormap = getattr(plt.cm, cmap)
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(uvals))]
        colors = np.random.permutation(colors)
        colormatch = {val: col for val, col in zip(uvals, colors)}
    elif type(cmap) == dict:
        colormatch = cmap
    else:
        raise Exception("`cmap` can only take a str or a dict")

    if isinstance(map_obj, mpl.collections.PolyCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_color([colormatch[i] for i in pvalues])
        map_obj.set_edgecolor('k')
    elif isinstance(map_obj, mpl.collections.LineCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_color([colormatch[i] for i in pvalues])
    elif isinstance(map_obj, mpl.collections.PathCollection):
        if not hasattr(map_obj, 'shp2dbf_row'):
            map_obj.shp2dbf_row = np.arange(values.shape[0])
        map_obj.set_array(values)
    return map_obj

def base_choropleth_classif(map_obj, values, classification='quantiles',
        k=5, cmap='hot_r', sample_fisher=False):
    '''
    Set coloring based based on different classification
    methods
    ...

    Arguments
    ---------

    map_obj         : Poly/Line collection
                      Output from map_X_shp
    values          : array
                      Numpy array with values to map
    classification  : str
                      Classificatio method to use. Options supported:
                        * 'quantiles' (default)
                        * 'fisher_jenks'
                        * 'equal_interval'

    k               : int
                      Number of bins to classify values in and assign a color
                      to
    cmap            : str
                      Matplotlib coloring scheme
    sample_fisher   : Boolean
                      Defaults to False, controls whether Fisher-Jenks
                      classification uses a sample (faster) or the entire
                      array of values. Ignored if 'classification'!='fisher_jenks'
                      The percentage of the sample that takes at a time is 10%

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    '''
    if classification == 'quantiles':
        classification = ps.Quantiles(values, k)
        boundaries = classification.bins.tolist()

    if classification == 'equal_interval':
        classification = ps.Equal_Interval(values, k)
        boundaries = classification.bins.tolist()

    if classification == 'fisher_jenks':
        if sample_fisher:
            classification = ps.esda.mapclassify.Fisher_Jenks_Sampled(values,k)
        else:
            classification = ps.Fisher_Jenks(values,k)
        boundaries = classification.bins[:]

    map_obj.set_alpha(0.4)

    cmap = cm.get_cmap(cmap, k+1)
    map_obj.set_cmap(cmap)

    boundaries = np.insert(boundaries, 0, values.min())
    norm = clrs.BoundaryNorm(boundaries, cmap.N)
    map_obj.set_norm(norm)

    if isinstance(map_obj, mpl.collections.PolyCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_array(pvalues)
        map_obj.set_edgecolor('k')
    elif isinstance(map_obj, mpl.collections.LineCollection):
        pvalues = _expand_values(values, map_obj.shp2dbf_row)
        map_obj.set_array(pvalues)
    elif isinstance(map_obj, mpl.collections.PathCollection):
        if not hasattr(map_obj, 'shp2dbf_row'):
            map_obj.shp2dbf_row = np.arange(values.shape[0])
        map_obj.set_array(values)
    return map_obj

def base_lisa_cluster(map_obj, lisa, p_thres=0.01):
    '''
    Set coloring on a map object based on LISA results
    ...

    Arguments
    ---------

    map_obj         : Poly/Line collection
                      Output from map_X_shp
    lisa            : Moran_Local
                      LISA object  from PySAL
    p_thres         : float
                      Significant threshold for clusters

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    '''
    sign = lisa.p_sim < p_thres
    quadS = lisa.q * sign
    sig_quadS = pd.Series(quadS).values
    lisa_patch = base_choropleth_unique(map_obj, sig_quadS, lisa_clrs)
    lisa_patch.set_alpha(1)
    return lisa_patch

def lisa_legend_components(lisa, p_thres):
    '''
    Generate the lists `boxes` and `labels` required to build LISA legend

    NOTE: if non-significant values, they're consistently assigned at the end
    ...

    Arguments
    ---------
    lisa            : Moran_Local
                      LISA object  from PySAL
    p_thres         : float
                      Significant threshold for clusters

    Returns
    -------
    boxes           : list
                      List with colors of the boxes to draw on the legend
    labels          : list
                      List with labels to anotate the legend colors, aligned
                      with `boxes`
    '''
    sign = lisa.p_sim < p_thres
    quadS = lisa.q * sign
    cls = list(set(quadS))
    boxes = []
    labels = []
    np.sort(cls)
    for cl in cls:
        boxes.append(mpl.patches.Rectangle((0, 0), 1, 1,
            facecolor=lisa_clrs[cl]))
        labels.append(lisa_lbls[cl])
    if 0 in cls:
        i = labels.index('Non-significant')
        boxes = boxes[:i] + boxes[i+1:] + [boxes[i]]
        labels = labels[:i] + labels[i+1:] + [labels[i]]
    return boxes, labels

def _expand_values(values, shp2dbf_row):
    '''
    Expand series of values based on dbf order to polygons (to allow plotting
    of multi-part polygons).
    ...

    NOTE: this is done externally so it's easy to drop dependency on Pandas
    when neccesary/time is available.

    Arguments
    ---------
    values          : ndarray
                      Values aligned with dbf rows to be plotted (e.d.
                      choropleth)
    shp2dbf_row    : list/sequence
                      Cardinality list of polygon to dbf row as provided by
                      map_poly_shp

    Returns
    -------
    pvalues         : ndarray
                      Values repeated enough times in the right order to be
                      passed from dbf to polygons
    '''
    pvalues = pd.Series(values, index=np.arange(values.shape[0]))\
            .reindex(shp2dbf_row)#Expand values to every poly
    return pvalues.values

# High-level pieces


def plot_poly_lines(shp_link,  savein=None, poly_col='none'):
    '''
    Quick plotting of shapefiles
    ...

    Arguments
    ---------
    shp_link        : str
                      Path to shapefile
    savein          : str
                      Path to png file where to dump the plot. Optional,
                      defaults to None
    poly_col        : str
                      Face color of polygons
    '''
    fig = plt.figure()
    shp = ps.open(shp_link)
    patchco = map_poly_shp(shp)
    patchco.set_facecolor('none')
    patchco.set_edgecolor('0.8')
    ax = setup_ax([patchco], [shp.bbox])
    fig.add_axes(ax)

    if savein:
        plt.savefig(savein)
    else:
        print('callng plt.show()')
        plt.show()
    return None

def plot_choropleth(shp_link, values, type, k=5, cmap=None,
        shp_type='poly', sample_fisher=False, title='',
        savein=None, figsize=None, dpi=300, alpha=0.4):
    '''
    Wrapper to quickly create and plot from a lat/lon shapefile
    ...

    Arguments
    ---------

    shp_link        : str
                      Path to shapefile
    values          : array
                      Numpy array with values to map
    type            : str
                      Type of choropleth. Supported methods:
                        * 'classless'
                        * 'unique_values'
                        * 'quantiles'
                        * 'fisher_jenks'
                        * 'equal_interval'
    k               : int
                      Number of bins to classify values in and assign a color
                      to (defaults to 5)
    cmap            : str
                      Matplotlib coloring scheme. If None (default), uses:
                        * 'classless': 'Greys'
                        * 'unique_values': 'Paired'
                        * 'quantiles': 'hot_r'
                        * 'fisher_jenks': 'hot_r'
                        * 'equal_interval': 'hot_r'
    shp_type        : str
                      'poly' (default) or 'line', for the kind of shapefile
                      passed
    sample_fisher   : Boolean
                      Defaults to False, controls whether Fisher-Jenks
                      classification uses a sample (faster) or the entire
                      array of values. Ignored if 'classification'!='fisher_jenks'
                      The percentage of the sample that takes at a time is 10%
    title           : str
                      Optional string for the title
    savein          : str
                      Path to png file where to dump the plot. Optional,
                      defaults to None
    figsize         : tuple
                      Figure dimensions
    dpi             : int
                      resolution of graphic file
    alpha           : float
                      [Optional. Default=0.4] Transparency of the map.

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    '''
    shp = ps.open(shp_link)
    if shp_type == 'poly':
        map_obj = map_poly_shp(shp)
    if shp_type == 'line':
        map_obj = map_line_shp(shp)

    if type == 'classless':
        if not cmap:
            cmap = 'Greys'
        map_obj = base_choropleth_classless(map_obj, values, cmap=cmap)
    if type == 'unique_values':
        if not cmap:
            cmap = 'Paired'
        map_obj = base_choropleth_unique(map_obj, values, cmap=cmap)
    if type == 'quantiles':
        if not cmap:
            cmap = 'hot_r'
        map_obj = base_choropleth_classif(map_obj, values, k=k, \
                classification='quantiles', cmap=cmap)
    if type == 'fisher_jenks':
        if not cmap:
            cmap = 'hot_r'
        map_obj = base_choropleth_classif(map_obj, values, k=k, \
                classification='fisher_jenks', cmap=cmap, \
                sample_fisher=sample_fisher)
    if type == 'equal_interval':
        if not cmap:
            cmap = 'hot_r'
        map_obj = base_choropleth_classif(map_obj, values, k=k, \
                classification='equal_interval', cmap=cmap)

    map_obj.set_alpha(alpha)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax = setup_ax([map_obj], [shp.bbox], ax)
    if title:
        ax.set_title(title)
    if type=='quantiles' or type=='fisher_jenks' or type=='equal_interval':
        cmap = map_obj.get_cmap()
        norm = map_obj.norm
        boundaries = np.round(map_obj.norm.boundaries, decimals=3)
        cbar = plt.colorbar(map_obj, cmap=cmap, norm=norm, boundaries=boundaries, \
                ticks=boundaries, orientation='horizontal', shrink=0.5)
    if savein:
        plt.savefig(savein, dpi=dpi)
    else:
        plt.show()
    return None

# Coding to be used with PySAL scheme
# HH=1, LH=2, LL=3, HL=4
lisa_clrs = {1: '#FF0000', 2: '#66CCFF', 3: '#003399', 4: '#CD5C5C', \
             0: '#D3D3D3'}
lisa_lbls = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL', \
             0: 'Non-significant'}

def plot_lisa_cluster(shp_link, lisa, p_thres=0.01, shp_type='poly',
        title='', legend=True, savein=None, figsize=None, dpi=300, alpha=1.,
        leg_loc=0):
    '''
    Plot LISA cluster maps easily
    ...

    Arguments
    ---------

    shp_link        : str
                      Path to shapefile
    lisa            : Moran_Local
                      LISA object  from PySAL. NOTE: assumes
                      `geoda_quads=False`
    p_thres         : float
                      Significant threshold for clusters
    shp_type        : str
                      'poly' (default) or 'line', for the kind of shapefile
                      passed
    title           : str
                      Optional string for the title
    legend          : Boolean
                      [Optional. Default=True] Flag to add a legend to the map
    savein          : str
                      Path to png file where to dump the plot. Optional,
                      defaults to None
    figsize         : tuple
                      Figure dimensions
    dpi             : int
                      resolution of graphic file
    alpha           : float
                      [Optional. Default=0.4] Transparency of the map.
    leg_loc         : int
                      [Optional. Default=0] Location of legend. 0: best, 1:
                      upper right, 2: upper left, 3: lower left, 4: lower
                      right, 5: right, 6: center left, 7: center right, 8: lower
                      center, 9: upper center, 10: center.

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    '''
    shp = ps.open(shp_link)
    # Lisa layer
    lisa_obj = map_poly_shp(shp)
    lisa_obj = base_lisa_cluster(lisa_obj, lisa)
    lisa_obj.set_alpha(alpha)
    # Figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax = setup_ax([lisa_obj], [shp.bbox], ax)
    # Legend
    if legend:
        boxes, labels = lisa_legend_components(lisa, p_thres)
        plt.legend(boxes, labels, loc=leg_loc, fancybox=True)
    if title:
        ax.set_title(title)
    if savein:
        plt.savefig(savein, dpi=dpi)
    else:
        plt.show()
    return None


if __name__ == '__main__':

    data = 'none'
    if data == 'poly':
        shp_link = ps.examples.get_path("sids2.shp")
        shp_link = ps.examples.get_path("Polygon.shp")
        dbf = ps.open(shp_link.replace('.shp', '.dbf'))
        '''
        values = np.array(dbf.by_col("SIDR74"))
        #values[: values.shape[0]/2] = 1
        #values[values.shape[0]/2: ] = 0
        '''
        patchco = map_poly_shp(ps.open(shp_link))
        #patchco = base_choropleth_classif(shp_link, np.random.random(3))
        #patchco = plot_choropleth(shp_link, np.random.random(3), 'quantiles')

    if data == 'point':
        shp_link = ps.examples.get_path("burkitt.shp")
        dbf = ps.open(shp_link.replace('.shp', '.dbf'))
        patchco = map_point_shp(ps.open(shp_link))

    if data == 'line':
        shp_link = ps.examples.get_path("eberly_net.shp")
        dbf = ps.open(shp_link.replace('.shp', '.dbf'))
        values = np.array(dbf.by_col('TNODE'))
        mobj = map_line_shp(ps.open(shp_link))
        patchco = base_choropleth_unique(mobj, values)

    '''
    which = values > 1.

    for shp_link in [shp_link]:

        fig = plt.figure()
        patchco = map_poly_shp(shp_link)
        patchcoB = map_poly_shp(shp_link, which=which)
        patchco.set_facecolor('none')
        ax = setup_ax([patchco, patchcoB])
        fig.add_axes(ax)
        plt.show()
        break

    xy = (((0, 0), (0, 0)), ((2, 1), (2, 1)), ((3, 1), (3, 1)), ((2, 5), (2, 5)))
    xy = np.array([[10, 30], [20, 20]])
    markerobj = mpl.markers.MarkerStyle('o')
    path = markerobj.get_path().transformed(
            markerobj.get_transform())
    scales = np.array([2, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pc = PathCollection((path,), scales, offsets=xy, \
            facecolors='r', transOffset=mpl.transforms.IdentityTransform())
    #pc.set_transform(mpl.transforms.IdentityTransform())
    #_ = _add_axes2col(pc, [0, 0, 5, 5])
    ax.add_collection(pc)
    fig.add_axes(ax)
    #ax = setup_ax([pc], ax)
    plt.show()
    '''

    shp_link = ps.examples.get_path('columbus.shp')
    values = np.array(ps.open(ps.examples.get_path('columbus.dbf')).by_col('HOVAL'))
    w = ps.queen_from_shapefile(shp_link)
    lisa = ps.Moran_Local(values, w, permutations=999)
    _ = plot_lisa_cluster(shp_link, lisa)
    #_ = plot_choropleth(shp_link, values, 'fisher_jenks')
