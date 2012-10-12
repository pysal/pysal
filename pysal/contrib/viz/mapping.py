"""
mapping.py

Illustration of how to do choropleth mapping by combining matplotlib and 
PySAL


"""

__author__ = "Sergio Rey <sjsrey@gmail.com>"


import pysal as ps
import numpy as np
import  matplotlib.pyplot as plt 
from matplotlib import colors as clrs
from matplotlib import mpl
from matplotlib.pyplot import fill, text
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

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
            boundaries)
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
    pass

