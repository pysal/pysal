"""
Contiguity based spatial weights.

"""

__author__ = "Sergio J. Rey <srey@asu.edu> "
__all__ = ['buildContiguity']

import pysal
from _contW_binning import ContiguityWeights_binning as ContiguityWeights
from _contW_binning import ContiguityWeightsPolygons


WT_TYPE = {'rook': 2, 'queen': 1}  # for _contW_Binning


def buildContiguity(polygons, criterion="rook", ids=None):
    """
    Build contiguity weights from a source.

    Parameters
    ----------

    polygons   :
                 an instance of a pysal geo file handler
                 Any thing returned by pysal.open that is explicitly polygons
    criterion  : string
                 contiguity criterion ("rook","queen")
    ids        : list
                 identifiers for i,j

    Returns
    -------

    w         : W
                instance; Contiguity weights object

    Examples
    --------

    >>> w = buildContiguity(pysal.open(pysal.examples.get_path('10740.shp'),'r'))
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w[0]
    {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
    >>> w = buildContiguity(pysal.open(pysal.examples.get_path('10740.shp'),'r'),criterion='queen')
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w.pct_nonzero
    3.1926364234056543
    >>> w = buildContiguity(pysal.open(pysal.examples.get_path('10740.shp'),'r'),criterion='rook')
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [163]
    >>> w.pct_nonzero
    2.6351084812623276
    >>> fips = pysal.open(pysal.examples.get_path('10740.dbf')).by_col('STFID')
    >>> w = buildContiguity(pysal.open(pysal.examples.get_path('10740.shp'),'r'),ids=fips)
    WARNING: there is one disconnected observation (no neighbors)
    Island id:  [u'35043940300']
    >>> w['35001000107']
    {u'35001003805': 1.0, u'35001003721': 1.0, u'35001000111': 1.0, u'35001000112': 1.0, u'35001000108': 1.0}

    Notes
    -----

    The types of sources supported will expand over time.

    See Also
    --------
    pysal.weights.W # need to fix sphinx links

    """

    if ids and len(ids) != len(set(ids)):
        raise ValueError("The argument to the ids parameter contains duplicate entries.")

    wt_type = WT_TYPE[criterion.lower()]
    geo = polygons
    if issubclass(type(geo), pysal.open):
        geo.seek(0)  # Make sure we read from the beginging of the file.
        geoObj = geo
    else:
        raise TypeError(
            "Argument must be a FileIO handler or connection string.")
    neighbor_data = ContiguityWeights(geoObj, wt_type).w
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
    return pysal.weights.W(neighbors, id_order=ids)

