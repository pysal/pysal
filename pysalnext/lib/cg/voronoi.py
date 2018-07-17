"""
Voronoi tesslation of 2-d point sets


Adapted from https://gist.github.com/pv/8036995

"""
import numpy as np
from scipy.spatial import Voronoi


__author__ = "Serge Rey <sjsrey@gmail.com>"

__all__ = ['voronoi']

def voronoi(points, radius=None):
    """
    Determine finite Voronoi diagram for a 2-d point set 


    Parameters
    ----------
    points      : array-like
                  nx2 array of points

    radius      : float (optional) 
                  distance to 'points at infinity'

    Returns
    -------

    regions    : list
                  each element of the list contains sequence of the indexes of Voronoi vertices composing a Voronoi polygon (region)

    coordinates : array
                  coordinates of the Voronoi vertices


    Examples
    --------
    >>> points = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]
    >>> regions, coordinates = voronoi(points)
    >>> regions
    [[0, 3, 4, 2], [1, 2, 5, 6], [2, 0, 1], [7, 0, 1, 8]]
    >>> coordinates
    array([[  6.09488636,  -8.11676136],
           [  3.82047478,   6.66691395],
           [  8.02880572,   7.67691337],
           [  5.69502851, -23.11143087],
           [ 15.39400167,  20.74419651],
           [ 15.39400167,  20.74419651],
           [ -8.52771701,  15.18290828],
           [  5.69502851, -23.11143087],
           [ -8.52771701,  15.18290828]])
    """
    vor = Voronoi(points)
    return voronoi_regions(vor, radius=radius)

def voronoi_regions(vor, radius=None):
    """
    Finite voronoi regions for a 2-d point set.


    Parameters
    ----------

    vor:  Voronoi (scipy.spatial)

    radius: float (optional)
            Distance to 'points at infinity'
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1] 
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def as_dataframes(regions, vertices, points):
    """
    Helper function to store finite Voronoi regions and originator points as
    geopandas (or pandas) data frames


    Parameters
    ----------

    regions     : list
                  each element of the list contains sequence of the indexes of
                  voronoi vertices composing a vornoi polygon (region)

    vertices    : array
                  coordinates of the vornoi vertices

    points      : array-like
                  originator points


    Returns
    -------

    region_df   : GeoDataFrame
                  Finite Voronoi polygons as geometries

    points_df   : GeoDataFrame
                  Originator points as geometries
    """
    try:
        import geopandas as gpd
    except ImportError:
        gpd = None

    try:
        from shapely.geometry import Polygon, Point
    except ImportError:
        from .shapes import Polygon, Point

    if gpd is not None:
        region_df = gpd.GeoDataFrame()
        region_df['geometry'] = [Polygon(vertices[region]) for region in regions]

        point_df = gpd.GeoDataFrame()
        point_df['geometry'] = gpd.GeoSeries(Point(pnt) for pnt in points)
    else:
        import pandas as pd
        region_df = pd.DataFrame()
        region_df['geometry'] = [Polygon(vertices[region].tolist()) for region in regions]
        point_df = pd.DataFrame()
        point_df['geometry'] = [Point(pnt) for pnt in points]

    return region_df, point_df

def voronoi_frames(points, radius=None):
    """
    Composite helper to return Voronoi regions and generator points as individual dataframes

    Parameters
    ----------

    points      : array-like
                  originator points


    Returns
    -------

    _           : tuple
                  (region_df, points_df)

                  region_df   : GeoDataFrame (if geopandas available, otherwise Pandas DataFrame)
                                Finite Voronoi polygons as geometries

                  points_df   : GeoDataFrame (if geopandas available, otherwise Pandas DataFrame)
                                Originator points as geometries

    Notes
    -----

    If Geopandas is not available the return types will be Pandas DataFrames
    each with a geometry column populated with PySAL shapes. If Geopandas is
    available, return types are GeoDataFrames with a geometry column populated
    with shapely geometry types.

    Examples
    --------
    >>> eoints = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]
    >>> regions_df, points_df = voronoi_frames(points)
    >>> regions_df.shape
    (4, 1)
    >>> regions_df.shape === points_df.shape
    True

    """
    regions, vertices = voronoi(points, radius=radius)
    return as_dataframes(regions, vertices, points)
