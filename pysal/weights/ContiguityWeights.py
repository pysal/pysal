import pysal
import numpy as num
import scipy as sci
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
    return _make_weights(geo,QUEEN)
def rook(geo):
    return _make_weights(geo,ROOK)

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
    r = rook('../examples/10740.shp')
    q = queen('../examples/10740.shp')

    


