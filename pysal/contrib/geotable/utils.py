from pysal.cg import asShape as pShape
from shapely.geometry import asShape as sShape
from shapely.geometry.base import BaseGeometry as sBaseGeometry
from geopandas import GeoDataFrame, GeoSeries
import pandas as pd
from functools import wraps
from pysal import open as popen
from warnings import warn

def to_df(df, geom_col='geometry', **kw):
    """
    Convert a Geopandas dataframe into a normal pandas dataframe with a column
    containing PySAL shapes. 

    Arguments
    ---------
    df      :   geopandas.GeoDataFrame
                a geopandas dataframe (or pandas dataframe) with a column
                containing geo-interfaced shapes
    geom_col:   str
                string denoting which column in the df contains the geometry
    **kw    :   keyword options
                options passed directly to pandas.DataFrame(...,**kw)

    See Also
    --------
    pandas.DataFrame
    """
    df[geom_col] = df[geom_col].apply(pShape)
    if isinstance(df, (GeoDataFrame, GeoSeries)):
        df = pd.DataFrame(df, **kw)
    return df

def to_gdf(df, geom_col='geometry', **kw):
    """
    Convert a pandas dataframe with geometry column to a GeoPandas dataframe

    Arguments
    ---------
    df      :   pandas.DataFrame
                a pandas dataframe with a column containing geo-interfaced
                shapes
    geom_col:   str
                string denoting which column in the df contains the geometry
    **kw    :   keyword options
                options passed directly to geopandas.GeoDataFrame(...,**kw)

    See Also
    --------
    geopandas.GeoDataFrame
    """
    df[geom_col] = df[geom_col].apply(sShape)
    return GeoDataFrame(df, geometry=geom_col, **kw)

def insert_metadata(df, obj, name=None, inplace=True, overwrite=False):
    if not inplace:
        new = df.copy(deep=True)
        insert_metadata(new, obj, name=name, inplace=True)
        return new
    if name is None:
        name = type(obj).__name__
    if hasattr(df, name):
        if overwrite:
            warn('Overwriting attribute {}! This may break the dataframe!'.format(name))
        else:
            raise Exception('Dataframe already has attribute {}. Cowardly refusing '
                        'to break dataframe. '.format(name))
    df._metadata.append(name)
    df.__setattr__(name, obj) 
