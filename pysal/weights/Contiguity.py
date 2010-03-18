"""
Contiguity based spatial weights

Author(s):
    Serge Rey srey@asu.edu
"""

import pysal
try:
    from _contW_rtree import ContiguityWeights_rtree as ContiguityWeights
except:
    from _contW_binning import ContiguityWeights_binning as ContiguityWeights


WT_TYPE={'rook':2,'queen':1} # for _contW_Binning

def buildContiguity(source,criterion="rook",ids=None):
    """
    Build contiguity weights from a source

    Parameters
    ----------

    source     : multitype 
                 polygon shapefile

    criterion   : string
                 contiguity criterion ("rook","queen")

    ids        : list
                 identifiers for i,j


    Returns
    -------

    w         : W instance
                Contiguity weights object


    Examples
    -------
    >>> w = buildContiguity('../examples/10740.shp')
    >>> w[0]
    {1: 1.0, 4: 1.0, 101: 1.0, 85: 1.0, 5: 1.0}
    >>> w = buildContiguity('../examples/10740.shp',criterion='queen')
    >>> w.pct_nonzero
    0.031926364234056544
    >>> w = buildContiguity('../examples/10740.shp',criterion='rook')
    >>> w.pct_nonzero
    0.026351084812623275


    Notes
    -----

    The types of sources supported will expand over time.

    See Also
    --------
    pysal.weights.W # need to fix sphinx links

    """
    
    wt_type=WT_TYPE[criterion.lower()]
    geo=source
    if issubclass(type(geo),basestring):
        geoObj = pysal.open(geo)
    elif issubclass(type(geo),pysal.open):
        geo.seek(0)
        geoObj = geo
    else:
        raise TypeError, "Argument must be a FileIO handler or connection string"
    neighbor_data = ContiguityWeights(geoObj,wt_type).w
    neighbors={}
    weights={}
    for key in neighbor_data:
        neighbors[key] = list(neighbor_data[key])
    return pysal.weights.W(neighbors,id_order=ids)

    
if __name__ == "__main__":

    import doctest
    doctest.testmod()
