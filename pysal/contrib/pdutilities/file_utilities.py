import pysal as ps
import os
from shp_utilities import shp2series, series2shp
from dbf_utilities import dbf2df, df2dbf


def read_files(filepath, **kwargs):
    """
    Reads a dbf/shapefile pair, squashing geometries into a "geometry" column.
    """
    #keyword arguments wrapper will strip all around dbf2df's required arguments
    geomcol = kwargs.pop('geomcol', 'geometry')
    weights = kwargs.pop('weights', '')
    
    dbf_path, shp_path = _pairpath(filepath)

    df = dbf2df(dbf_path, **kwargs)
    df[geomcol] = shp2series(shp_path)

    if weights != '' and isinstance(weights, str):
        if weights.lower() in ['rook', 'queen']:
            if weights.lower() == 'rook':
                df.W = ps.rook_from_shapefile(shp_path)
            else:
                df.W = ps.queen_from_shapefile(shp_path)
        else:
            try:
                W_path = os.path.splitext(dbf_path)[0] + '.' + weights
                df.W = ps.open(W_path).read()
            except IOError:
                print('Weights construction failed! Passing on weights')
    
    return df
   
def write_files(df, filepath, **kwargs):
    """
    writes dataframes with potential geometric components out to files. 
    """
    geomcol = kwargs.pop('geomcol', 'geometry')
    weights = kwargs.pop('weights', 'gal')
    
    dbf_path, shp_path = _pairpath(filepath)

    if geomcol not in df.columns:
        return df2dbf(df, dbf_path, **kwargs)
    else:
        shp_out = series2shp(df[geomcol], shp_path)
        not_geom = [x for x in df.columns if x != geomcol]
        dbf_out = df2dbf(df[not_geom], dbf_path, **kwargs)
        if hasattr(df, 'W'):
            W_path = os.path.splitext(filepath)[0] + '.' + weights
            ps.open(W_path, 'w').write(df.W)
        else:
            W_path = 'no weights written'
        return dbf_out, shp_out, W_path
            


def _pairpath(filepath):
    """
    return dbf/shp paths for any shp, dbf, or basepath passed to function. 
    """
    base = os.path.splitext(filepath)[0]
    return base + '.dbf', base + '.shp'
    
