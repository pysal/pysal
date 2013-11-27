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
try:
    from mpl_toolkits.basemap import Basemap
except:
    print "WARNING: Basemap not installed and cannot be imported"
try:
    from ogr import osr
except:
    print "WARNING: 'ogr' could not be imported. Reprojections won't work"

def transCRS(xy, src_prj, trt_prj):
    '''
    Re-project a 2D array of xy coordinates from one prj file to another
    ...

    Arguments
    ---------
    xy          : ndarray
                  nx2 array with coordinates to be reprojected. First column
                  is X axis, second is Y axis
    src_prj     : str
                  Path to .prj file of the source Coordinate Reference System
                  (CRS) of `xy`
    trt_prj     : str
                  Path to .prj file of the target Coordinate Reference System
                  (CRS) to reproject `xy`

    Returns
    -------
    xyp         : ndarray
                  nx2 array with reprojected coordinates. First column
                  is X axis, second is Y axis

    '''
    orig = osr.SpatialReference()
    orig.ImportFromWkt(open(src_prj).read())
    target = osr.SpatialReference()
    target.ImportFromWkt(open(trt_prj).read())
    trCRS = osr.CoordinateTransformation(orig, target)
    return np.array(trCRS.TransformPoints(xy))[:, :2]

def map_point_shp(shp, which='all'):
    '''
    Create a map object from a point shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL point iterable with the attribute `bbox` (e.g.
                      shape object from `ps.open` a poly shapefile)
    which           : str/list

    Returns
    -------

    map             : PatchCollection
                      Map object with the points from the shape

    '''
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
    _ = _add_axes2col(sc, shp.bbox)
    return sc

def map_line_shp(shp, which='all'):
    '''
    Create a map object from a line shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL line iterable with the attribute `bbox` (e.g.
                      shape object from `ps.open` a poly shapefile)
    which           : str/list

    Returns
    -------

    map             : PatchCollection
                      Map object with the lines from the shape
                      This includes the attribute `shp2dbf_row` with the
                      cardinality of every line to its row in the dbf
                      (zero-offset)

    '''
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
    _ = _add_axes2col(lc, shp.bbox)
    lc.shp2dbf_row = rows
    return lc

def map_poly_shp(shp, which='all'):
    '''
    Create a map object from a polygon shape
    ...

    Arguments
    ---------

    shp             : iterable
                      PySAL polygon iterable with the attribute `bbox` (e.g.
                      shape object from `ps.open` a poly shapefile)
    which           : str/list
                      List of booleans for which polygons of the shapefile to
                      be included (True) or excluded (False)

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shape
                      This includes the attribute `shp2dbf_row` with the
                      cardinality of every polygon to its row in the dbf
                      (zero-offset)

    '''
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
    _ = _add_axes2col(pc, shp.bbox)
    pc.shp2dbf_row = rows
    return pc

def map_poly_shp_lonlat(shp_link, projection='merc'):
    '''
    Create a map object from a shapefile in lon/lat CRS using Basemap

    NOTE: deprecated in higher level functions for dependency on Basemap
    ...

    Arguments
    ---------

    shp_link        : str
                      Path to shapefile
    projection      : str
                      Basemap projection. See [1]_ for a list. Defaults to
                      'merc'

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile

    Links
    -----
    .. [1] <http://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>
    '''
    shp = ps.open(shp_link)
    shps = list(shp)
    left, bottom, right, top = shp.bbox
    m = Basemap(resolution = 'i', projection=projection,
            llcrnrlat=bottom, urcrnrlat=top,
            llcrnrlon=left, urcrnrlon=right,
            lat_ts=(bottom+top)/2,
            lon_0=(right-left)/2, lat_0=(top-bottom)/2)
    bounding_box = [m.llcrnrx, m.llcrnry,m.urcrnrx,m.urcrnry]
    patches = []
    for shape in shps:
        parts = []
        for ring in shape.parts:
            xy = np.array(ring)
            x,y = m(xy[:,0], xy[:,1])
            x = x / bounding_box[2]
            y = y / bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            xy = np.hstack((x,y))
            polygon = Polygon(xy, True)
            patches.append(polygon)
    return PatchCollection(patches)

def setup_ax(polyCos_list, ax=None):
    '''
    Generate an Axes object for a list of collections
    ...

    Arguments
    ---------
    polyCos_list: list
                  List of Matplotlib collections (e.g. an object from
                  map_poly_shp)
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
    # Determine bboxes of new axes
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    for polyCo in polyCos_list:
        axs = polyCo.get_axes()
        xmin, xmax = axs.get_xlim()
        ymin, ymax = axs.get_ylim()
        if xmin < xlim[0]:
            xlim[0] = xmin
        if xmax > xlim[1]:
            xlim[1] = xmax
        if ymin < ylim[0]:
            ylim[0] = ymin
        if ymax > ylim[1]:
            ylim[1] = ymax
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Resize bbox of each coll and add it to axes
    for polyCo in polyCos_list:
        polyCo.get_axes().set_xlim(ax.get_xlim())
        polyCo.get_axes().set_ylim(ax.get_ylim())
        ax.add_collection(polyCo)
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

def plot_poly_lines(shp_link, projection='merc', savein=None, poly_col='none'):
    '''
    Quick plotting of shapefiles
    ...

    Arguments
    ---------
    shp_link        : str
                      Path to shapefile
    projection      : str
                      Basemap projection. See [1]_ for a list. Defaults to
                      'merc'
    savein          : str
                      Path to png file where to dump the plot. Optional,
                      defaults to None
    poly_col        : str
                      Face color of polygons
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    shp = ps.open(shp_link)
    patchco = map_poly_shp(shp)
    patchco.set_facecolor('none')
    patchco.set_edgecolor('0.8')
    ax = setup_ax([patchco], ax)
    if savein:
        plt.savefig(savein)
    else:
        plt.show()
    return None

def plot_choropleth(shp_link, values, type, k=5, cmap=None, \
        shp_type='poly', sample_fisher=True, title='', \
        savein=None, figsize=None, dpi=300):
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
                        * 'quantiles' (default)
                        * 'fisher_jenks'
                        * 'equal_interval'
    k               : int
                      Number of bins to classify values in and assign a color
                      to (defaults to 5)
    cmap            : str
                      Matplotlib coloring scheme. If None (default), uses:
                        * 'classless': 'Greys'
                        * 'unique_values': 'hot_r'
                        * 'quantiles': 'hot_r'
                        * 'fisher_jenks': 'hot_r'
                        * 'equal_interval': 'hot_r'
    shp_type        : str
                      'poly' (default) or 'line', for the kind of shapefile
                      passed
    sample_fisher   : Boolean
                      Defaults to True, controls whether Fisher-Jenks
                      classification uses a sample (faster) or the entire
                      array of values. Ignored if 'classification'!='fisher_jenks'
    title           : str
                      Optional string for the title
    savein          : str
                      Path to png file where to dump the plot. Optional,
                      defaults to None
    figsize         : tuple
                      Figure dimensions
    dpi             : int
                      resolution of graphic file

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    Links
    -----
    .. [1] <http://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>
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
            cmap = 'hot_r'
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

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax = setup_ax([map_obj], ax)
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


def base_choropleth_classless(map_obj, values, cmap='Greys', projection='merc'):
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
    projection      : str
                      Basemap projection. See [1]_ for a list. Defaults to
                      'merc'

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      classless coloring

    Links
    -----
    .. [1] <http://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>
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

def base_choropleth_unique(map_obj, values,  cmap='hot_r', projection='merc'):
    '''
    Set coloring based on unique values from a map object
    ...

    Arguments
    ---------

    map_obj         : Poly/Line collection
                      Output from map_X_shp
    values          : array
                      Numpy array with values to map
    cmap            : str
                      Matplotlib coloring scheme
    projection      : str
                      Basemap projection. See [1]_ for a list. Defaults to
                      'merc'

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    Links
    -----
    .. [1] <http://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>
    '''
    uvals = np.unique(values)
    colormap = getattr(plt.cm, cmap)
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(uvals))]
    colors = np.random.permutation(colors)
    colormatch = {val: col for val, col in zip(uvals, colors)}

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

def base_choropleth_classif(map_obj, values, classification='quantiles', \
        k=5, cmap='hot_r', sample_fisher=True):
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
                      Defaults to True, controls whether Fisher-Jenks
                      classification uses a sample (faster) or the entire
                      array of values. Ignored if 'classification'!='fisher_jenks'

    Returns
    -------

    map             : PatchCollection
                      Map object with the polygons from the shapefile and
                      unique value coloring

    Links
    -----
    .. [1] <http://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>
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

    boundaries.insert(0,0)
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

            #############################
            ### Serge's original code ###
            #############################

class Map_Projection(object):
    """Map_Projection

    Parameters
    ==========

    shapefile: name of shapefile with .shp extension

    projection: proj4 projection string


    Returns
    =======

    projected: list of lists
        projected coordinates for each shape in the shapefile. Each
        sublist contains projected coordinates for parts of a  shape


    """
    def __init__(self, shapefile, projection='merc'):
        super(Map_Projection, self).__init__()
        self.projection = projection
        shp_reader = ps.open(shapefile)
        shps = []
        for shp in shp_reader:
            shps.append(shp)
        left = shp_reader.header['BBOX Xmin']
        right = shp_reader.header['BBOX Xmax']
        bottom = shp_reader.header['BBOX Ymin']
        top = shp_reader.header['BBOX Ymax']
        m = Basemap(resolution = 'i', projection='merc',
                llcrnrlat=bottom, urcrnrlat=top,
                llcrnrlon=left, urcrnrlon=right,
                lat_ts=(bottom+top)/2)
        projected = []
        for shp in shps:
            parts = []
            for ring in shp.parts:
                xy = np.array(ring)
                x,y = m(xy[:,0], xy[:,1])
                parts.append([x,y])
            projected.append(parts)
        results = {}
        self.projected = projected
        self.bounding_box = [m.llcrnrx, m.llcrnry,m.urcrnrx,m.urcrnry]
        self.shapefile = shapefile

def equal_interval_map(coords, y, k, title='Equal Interval'):
    """

    coords: Map_Projection instance

    y: array
       variable to map

    k: int
       number of classes

    title: string
           map title
    """
    classification = ps.Equal_Interval(y,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patches = []
    colors = []
    i = 0
    shape_colors = classification.bins[classification.yb]
    shape_colors = y
    #classification.bins[classification.yb]
    for shp in coords.projected:
        for ring in shp:
            x,y = ring
            x = x / coords.bounding_box[2]
            y = y / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            xy = np.hstack((x,y))
            polygon = Polygon(xy, True)
            patches.append(polygon)
            colors.append(shape_colors[i])
        i += 1
    cmap = cm.get_cmap('hot_r', k+1)
    boundaries = classification.bins.tolist()
    boundaries.insert(0,0)
    norm = clrs.BoundaryNorm(boundaries, cmap.N)
    p = PatchCollection(patches, cmap=cmap, alpha=0.4, norm=norm)
    colors = np.array(colors)
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.colorbar(p, cmap=cmap, norm = norm, boundaries = boundaries, ticks=
            boundaries, shrink=0.5)
    plt.show()
    return classification


def fisher_jenks_map(coords, y, k, title='Fisher-Jenks', sampled=False):
    """

    coords: Map_Projection instance

    y: array
       variable to map

    k: int
       number of classes

    title: string
           map title

    sampled: binary
             if True classification bins obtained on a sample of y and then
                 applied. Useful for large n arrays
    """


    if sampled:
        classification = ps.esda.mapclassify.Fisher_Jenks_Sampled(y,k)
    else:
        classification = ps.Fisher_Jenks(y,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patches = []
    colors = []
    i = 0
    shape_colors = y
    #classification.bins[classification.yb]
    for shp in coords.projected:
        for ring in shp:
            x,y = ring
            x = x / coords.bounding_box[2]
            y = y / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            xy = np.hstack((x,y))
            polygon = Polygon(xy, True)
            patches.append(polygon)
            colors.append(shape_colors[i])
        i += 1
    cmap = cm.get_cmap('hot_r', k+1)
    boundaries = classification.bins[:]
    #print boundaries
    #print min(shape_colors) > 0.0
    if min(shape_colors) > 0.0:
        boundaries.insert(0,0)
    else:
        boundaries.insert(0, boundaries[0] - boundaries[1])
    #print boundaries
    norm = clrs.BoundaryNorm(boundaries, cmap.N)
    p = PatchCollection(patches, cmap=cmap, alpha=0.4, norm=norm)
    colors = np.array(colors)
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.colorbar(p, cmap=cmap, norm = norm, boundaries = boundaries, ticks=
             boundaries)
    plt.show()
    return classification



def quantile_map(coords,y,k, title='Quantile'):
    """
    Quantile choropleth map

    Arguments
    =========

    coords: Map_Projection instance

    y: array
       variable to map

    k: int
       number of classes

    title: string
           map title

    """


    classification = ps.Quantiles(y,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patches = []
    colors = []
    i = 0
    shape_colors = classification.bins[classification.yb]
    shape_colors = y
    #classification.bins[classification.yb]
    for shp in coords.projected:
        for ring in shp:
            x,y = ring
            x = x / coords.bounding_box[2]
            y = y / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            xy = np.hstack((x,y))
            polygon = Polygon(xy, True)
            patches.append(polygon)
            colors.append(shape_colors[i])
        i += 1
    cmap = cm.get_cmap('hot_r', k+1)
    boundaries = classification.bins.tolist()
    boundaries.insert(0,0)
    norm = clrs.BoundaryNorm(boundaries, cmap.N)
    p = PatchCollection(patches, cmap=cmap, alpha=0.4, norm=norm)
    colors = np.array(colors)
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.colorbar(p, cmap=cmap, norm = norm, boundaries = boundaries, ticks=
            boundaries)
    plt.show()
    return classification



def classless_map(coords,y, title='Classless'):
    """
    Classless choropleth map

    Arguments
    =========

    coords: Map_Projection instance

    y: array
       variable to map

    title: string
           map title

    """


    fig = plt.figure()
    ax = fig.add_subplot(111)
    patches = []
    colors = []
    i = 0
    shape_colors = y
    for shp in coords.projected:
        for ring in shp:
            x,y = ring
            x = x / coords.bounding_box[2]
            y = y / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            xy = np.hstack((x,y))
            polygon = Polygon(xy, True)
            patches.append(polygon)
            colors.append(shape_colors[i])
        i += 1
    cmap = cm.get_cmap('hot_r')
    p = PatchCollection(patches, cmap=cmap, alpha=0.4)
    colors = np.array(colors)
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.colorbar(p)
    plt.show()


def lisa_cluster_map(coords, lisa,  title='LISA Cluster Map', p = 0.05):
    """
    LISA Cluster Map

    Arguments
    =========

    coords: Map_Projection instance

    lisa: Moran_Local instance

    title: string
           map title

    p: float
       p-value to define clusters
    """

    # pysal: 1 HH,  2 LH,  3 LL,  4 HL
    c ={}
    c[0] = 'white' # non-significant
    c[1] = 'darkred'
    c[2] = 'lightsalmon'
    c[3] = 'darkblue'
    c[4] = 'lightblue'

    q = lisa.q.copy()
    yp = lisa.p_sim.copy()
    nsig = yp >  p
    q[nsig] = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    for shp in coords.projected:
        for ring in shp:
            x,y = ring
            x = x / coords.bounding_box[2]
            y = y / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            y.shape = (n,1)
            ax.fill(x,y,c[q[i]])
        i += 1
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.show()


def unique_values_map(coords,y, title='Unique Value'):
    """
    Unique value choropleth

    Arguments
    =========
    coords: Map_Projection instance

    y: array
       zeros for elements that should not be mapped, 1-4 for elements to
       highlight

    title: string
           map title


    Notes
    =====
    Allows for an unlimited number of categories, but if there are many
    categories the colors may be difficult to distinguish.
    [Currently designed for use with a Moran_Local Instance for mapping a
    subset of the significant LISAs.]

    """
    yu = np.unique(y)
    colormap = plt.cm.Set1
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(yu))]
    colors = np.random.permutation(colors)
    colormatch = zip(yu, colors)
    c = {}
    for i in colormatch:
        c[i[0]] = i[1]
    '''
    # pysal: 1 HH,  2 LH,  3 LL,  4 HL
    c ={}
    c[0] = 'white' # non-significant
    c[1] = 'darkred'
    c[2] = 'lightsalmon'
    c[3] = 'darkblue'
    c[4] = 'lightblue'
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    for shp in coords.projected:
        for ring in shp:
            x,yc = ring
            x = x / coords.bounding_box[2]
            yc = yc / coords.bounding_box[3]
            n = len(x)
            x.shape = (n,1)
            yc.shape = (n,1)
            ax.fill(x,yc,color=c[y[i]], edgecolor='black')
            #ax.fill(x,yc,c[y[i]])
        i += 1
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_title(title)
    plt.show()



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
    '''

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

