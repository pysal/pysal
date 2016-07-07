import pandas as pd
import pysal as ps

def shp2series(filepath):
    """
    reads a shapefile, stuffing each shape into an element of a Pandas Series
    """
    f = ps.open(filepath)
    s = pd.Series(poly for poly in f)
    f.close()
    return s

def series2shp(series, filepath):
    """
    writes a series of pysal polygons to a file
    """
    f = ps.open(filepath, 'w')
    for poly in series:
        f.write(poly)
    f.close()
    return filepath
