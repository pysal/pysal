from .weights import W
from pysal.cg import asShape
from pysal.core.FileIO import FileIO
from pysal.weights._contW_binning import ContiguityWeightsPolygons
from pysal.weights._contW_rtree import ContiguityWeights_rtree
WT_TYPE = {'rook': 2, 'queen': 1}  # for _contW_Binning

class Rook(W):
    def __init__(self, polygons, **kwargs):
        neighbors, ids = _build(polygons, criterion='rook',  **kwargs)
        W.__init__(self, neighbors, ids=ids, **kwargs)
    
    @classmethod
    def from_shapefile(cls, filepath, **kwargs):
        return cls(FileIO(filepath), **kwargs)
    
    @classmethod
    def from_iterable(cls, iterable, **kwargs):
        new_iterable = [asShape(shape) for shape in iterable]
        return cls(new_iterable, **kwargs)
    
    @classmethod
    def from_dataframe(cls, df, geomcol='geometry', **kwargs):
        return cls.from_iterable(df[geomcol].tolist(), **kwargs)

class Queen(W):
    def __init__(self, polygons, driver='binning', criterion='queen', **kw):
        neighbors, ids = _build(polygons, criterion=criterion, driver=driver)
        W.__init__(self, neighbors, ids=ids, **kw)
    
    @classmethod
    def from_shapefile(cls, filepath, **kwargs):
        return cls(FileIO(filepath), **kwargs)

    @classmethod
    def from_iterable(cls, iterable, **kwargs):
        new_iterable = [asShape(shape) for shape in iterable]
        return cls(new_iterable, **kwargs)

    @classmethod
    def from_dataframe(cls, df, geomcol='geometry', **kwargs):
        return cls.from_iterable(df[geomcol].tolist(), **kwargs)

def _build(polygons, criterion="rook", ids=None, driver='binning'):
    """
    This deviates from pysal.weights.buildContiguity in three ways:
    1. it picks a driver based on driver keyword
    2. it uses ContiguityWeightsPolygons
    3. it returns a neighbors, ids tuple, rather than instantiating a weights
    object
    """

    if ids and len(ids) != len(set(ids)):
        raise ValueError("The argument to the ids parameter contains duplicate entries.")

    wttype = WT_TYPE[criterion.lower()]
    geo = polygons
    if issubclass(type(geo), FileIO):
        geo.seek(0)  # Make sure we read from the beginning of the file.

    if driver.lower().startswith('rtree'):
        neighbor_data = ContiguityWeights_rtree(polygons, joinType=wttype).w
    else:
        neighbor_data = ContiguityWeightsPolygons(polygons, wttype=wttype).w

    neighbors = {}
    #weights={}
    if ids:
        for key in neighbor_data:
            ida = ids[key]
            if ida not in neighbors:
                neighbors[ida] = set()
            neighbors[ida].update([ids[x] for x in neighbor_data[key]])
        for key in neighbors:
            neighbors[key] = list(neighbors[key])
    else:
        for key in neighbor_data:
            neighbors[key] = list(neighbor_data[key])
    return neighbors, ids
