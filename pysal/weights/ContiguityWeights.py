import pysal

try:
    from _contW_rtree import ContiguityWeights_rtree as ContiguityWeights
except ImportError:
    from _contW_binning import ContiguityWeights_binning as ContiguityWeights

QUEEN = 1
ROOK = 2

def rook_from_shapefile(filename,id_variable):
    """ Arguments:
            filename: path to shapefile
            id_variable: Field name for the unique ID.
    """
    pass
def queen_from_shapefile(filename,id_variable):
    """ Arguments:
            filename: path to shapefile
            id_variable: Field name for the unique ID.
    """
    pass

def queen(geo):
    """queen W instance from a geo object

    Examples
    --------

    Replicate OpenGeoDa

    >>> w=queen('../examples/Chicago77.shp')
    >>> w.histogram
    [(1, 1), (2, 3), (3, 7), (4, 17), (5, 14), (6, 20), (7, 9), (8, 5), (9, 1)]
    """

    return _make_weights(geo,QUEEN)
def rook(geo):
    """rook W instance from a geo object

    Examples
    --------
    >>> w=shp_to_rook("../examples/10740.shp")
    >>> w.n
    195
    >>> w.pct_nonzero
    0.026351084812623275
    >>> 
    """
    return _make_weights(geo,ROOK)

def shp_to_rook(shape_file):
    """generate a rook contiguity matrix from a shapefile
    
    Examples
    --------
    >>> w=shp_to_rook("../examples/10740.shp")
    >>> w.n
    195
    >>> w.pct_nonzero
    0.026351084812623275
    >>> 
    
    """
    return rook(shape_file)

def _make_weights(geo,wt_type):
    if issubclass(type(geo),basestring):
        geoObj = pysal.open(geo)
    elif issubclass(type(geo),pysal.open):
        geo.seek(0)
        geoObj = geo
    else:
        raise TypeError, "Argument must be a FileIO handler or connection string"
    w = ContiguityWeights(geoObj,wt_type)
    return pysal.weights.W.fromBinary(w.w)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    


