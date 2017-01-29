from ...cg import asShape as pShape
from ...common import requires as _requires
from warnings import warn

@_requires('geopandas')
def to_df(df, geom_col='geometry', **kw):
    """
    Convert a Geopandas dataframe into a normal pandas dataframe with a column
    containing PySAL shapes. Always returns a copy. 

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
    out = df.copy(deep=True)
    out[geom_col] = out[geom_col].apply(pShape)
    return pd.DataFrame(out, **kw)

@_requires('geopandas')
def to_gdf(df, geom_col='geometry', **kw):
    """
    Convert a pandas dataframe with geometry column to a GeoPandas dataframe. Returns a copy always.

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
    out = df.copy(deep=True)
    out[geom_col] = out[geom_col].apply(sShape)
    out = GeoDataFrame(out, geometry=geom_col, **kw)
    return out

def insert_metadata(df, obj, name=None, inplace=False, overwrite=False):
    """
    Insert an object into a dataframe's metadata with a given key. 

    Arguments
    ------------
    df          : pd.DataFrame
                  dataframe to insert into the metadata
    obj         : object
                  object desired to insert into the dataframe
    name        : string
                  key of the object to use. Will be available as 
                  an attribute of the dataframe. 
    inplace     : bool
                  flag to denote whether to operate on a copy 
                  of the dataframe or not. 
    overwrite   : bool
                  flag to denote whether to replace existing entry
                  in metadata or not. 
    
    Returns
    --------
    If inplace, changes dataframe implicitly. 
    Else, returns a new dataframe with added metadata. 
    """
    if not inplace:
        new = df.copy(deep=True)
        insert_metadata(new, obj, name=name, 
                        inplace=True, overwrite=overwrite)
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
