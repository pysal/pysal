from ...cg.shapes import asShape as pShape
from ...common import requires as _requires
from warnings import warn

@_requires('geopandas')
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
    import pandas as pd
    from geopandas import GeoDataFrame, GeoSeries
    df[geom_col] = df[geom_col].apply(pShape)
    if isinstance(df, (GeoDataFrame, GeoSeries)):
        df = pd.DataFrame(df, **kw)
    return df

@_requires('geopandas')
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
    from geopandas import GeoDataFrame
    from shapely.geometry import asShape as sShape
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
