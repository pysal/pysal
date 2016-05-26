import numpy as np
from scipy.spatial import KDTree, cKDTree
from pysal.cg.kdtree import Arc_KDTree
import util
import pysal as ps
from pysal.common import intercept_filepath

KDTreeTypes = [KDTree, cKDTree, Arc_KDTree]

@intercept_filepath
def _W_Contiguity(collection, wtype='rook', **kwargs):
    """
    Arbitrary contiguity constructor from an iterable of polygons
    """
    WTYPE = wtype.upper()
    if hasattr(collection, 'geometry'):
        collection = collection.geometry
    if WTYPE not in ['QUEEN', 'ROOK']:
        raise ValueError("wtype must be 'QUEEN' or 'ROOK'")
    neighs = ps.weights.Contiguity.buildContiguity(collection, criterion=wtype)
    return neighs

@intercept_filepath
def _W_Distance(collection, constructor, *args, **kwargs):
    """
    Arbitrary distance-weights constructor from an iterable of shapes. 
    """
    if hasattr(collection, 'geometry'):
        collection = collection.geometry
    data = util.get_points_array(collection)
    return constructor(data, *args, **kwargs)

def Rook(collection, **kwargs):
    """
    Specific rook contiguity constructor from an iterable of polygons
    """
    return _W_Contiguity(collection, wtype='rook')

def Queen(collection, **kwargs):
    """
    Specific queen contiguity constructor from an iterable of polygons
    """
    return _W_Contiguity(collection, wtype='queen')

def Knn(collection, **kwargs):
    """
    K-nearest neighbors constructor from a container of points
    """
    return _W_Distance(collection, ps.weights.Distance.knnW, **kwargs)

def Kernel(collection, **kwargs):
    return _W_Distance(collection, ps.weights.Distance.Kernel, **kwargs)

def Kernel_Adaptive(collection, *args, **kwargs):
    kwargs['fixed'] = False
    kwargs['bandwidth'] = kwargs.pop('bandwidths', None)
    return _W_Distance(collection, ps.weights.Distance.Kernel, *args, **kwargs) 

def Threshold_Binary(collection, *args, **kwargs):
    return _W_Distance(collection, ps.weights.Distance.DistanceBand, *args, **kwargs)

def Distance_Band(collection, *args, **kwargs):
    return _W_Distance(collection, ps.weights.Distance.DistanceBand, *args, **kwargs)

def Threshold_Continuous(collection, *args, **kwargs):
    kwargs['binary'] = False
    return _W_Distance(collection, ps.weights.Distance.DistanceBand, *args, **kwargs)
