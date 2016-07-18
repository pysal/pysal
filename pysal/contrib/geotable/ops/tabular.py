from ....common import requires as _requires
from ..utils import to_gdf, to_df
from warnings import warn as _Warn
import functools as _f
import sys as _sys

@_requires("geopandas")
def spatial_join(df1, df2, left_geom_col='geometry', 
          right_geom_col='geometry', **kwargs):
    """
    Spatial join of two Pandas DataFrames. Calls out to Geopandas.

    Parameters
    ----------
    left_df : pandas.DataFrame
    right_df: pandas.DataFrames
    how     : string, default 'inner'
              The type of join:
              * 'left': use keys from left_df; retain only 
                left_df geometry column
              * 'right': use keys from right_df; retain only 
                right_df geometry column
              * 'inner': use intersection of keys from both dfs; 
                retain only left_df geometry column
    op      : string, default 'intersection'
              One of {'intersects', 'contains', 'within'}.
              See http://toblerity.org/shapely/manual.html#binary-predicates.
    lsuffix : string, default 'left'
              Suffix to apply to overlapping column names 
              (left GeoDataFrame).
    rsuffix : string, default 'right'
              Suffix to apply to overlapping column namei
              (right GeoDataFrame).
    """
    import geopandas as gpd
    gdf1 = to_gdf(df1, geom_col=left_geom_col)
    gdf2 = to_gdf(df2, geom_col=right_geom_col)
    out = gpd.tools.sjoin(gdf1, gdf2, **kwargs)
    return to_df(out)

try:
    import pandas as _pd
    @_requires("pandas")
    @_f.wraps(_pd.merge)
    def join(*args, **kwargs):
        return _pd.merge(*args, **kwargs)
except ImportError:
    pass

@_requires("geopandas")
def spatial_overlay(df1, df2, how, left_geom_col='geometry', 
            right_geom_col='geometry', **kwargs):
    """
    Perform spatial overlay between two polygonal datasets. 
    Calls out to geopandas.
    
    Currently only supports data pandas.DataFrames with polygons.
    Implements several methods that are all effectively subsets of
    the union.
    
    Parameters
    ----------
    df1 : pandas.DataFrame 
          must have MultiPolygon or Polygon geometry column
    df2 : pandas.DataFrame 
          must have MultiPolygon or Polygon geometry column
    how : string
          Method of spatial overlay: 'intersection', 'union',
          'identity', 'symmetric_difference' or 'difference'.
    use_sindex : boolean, default True
                 Use the spatial index to speed up operation 
                 if available.
    
    Returns
    -------
    df : pandas.DataFrame
    pandas.DataFrame with new set of polygons and attributes
    resulting from the overlay
    """
    import geopandas as gpd
    gdf1 = to_gdf(df1, geom_col=left_geom_col)
    gdf2 = to_gdf(df2, geom_col=right_geom_col)
    out = gpd.tools.overlay(gdf1, gdf2, how, **kwargs)
    return to_df(out)

@_requires('shapely')
def dissolve(df, by='', **groupby_kws):
    from ._shapely import cascaded_union as union
    return union(df, by=by, **groupby_kws)

def clip(return_exterior=False):
    # return modified entries of the df that are within an envelope
    # provide an option to null out the geometries instead of not returning
    raise NotImplementedError

def erase(return_interior=True):
    # return modified entries of the df that are outside of an envelope
    # provide an option to null out the geometries instead of not returning 
    raise NotImplementedError

@_requires('shapely')
def union(df, **kws):
    if 'by' in kws:
        warn('when a "by" argument is provided, you should be using "dissolve"') 
        return dissolve(df, **kws)
    from ._shapely import cascaded_union as union
    return union(df)

@_requires('shapely')
def intersection(df, **kws):
    from ._shapely import cascaded_intersection as intersection
    return intersection(df, **kws) 

def symmetric_difference():
    raise NotImplementedError

def difference():
    raise NotImplementedError
