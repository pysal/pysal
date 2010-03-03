"""
Convenience functions for the construction of spatial weights based on
contiguity criteria

Author(s):
    Serge Rey srey@asu.edu

"""
from Contiguity import buildContiguity

def queen_from_shapefile(shapefile):
    """
    Queen contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile : string
                name of shapefile including suffix.

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wq=queen_from_shapefile("../examples/columbus.shp")
    >>> wq.pct_nonzero
    0.098292378175760101


    Notes
    -----

    Queen contiguity defines as neighbors any pair of polygons that share at
    least one vertex in their polygon definitions.

    See Also
    --------

    pysal.weights.W
    """
    return buildContiguity(shapefile,criteria='queen')

def rook_from_shapefile(shapefile):
    """
    Rook contiguity weights from a polygon shapefile

    Parameters
    ----------

    shapefile : string
                 name of shapefile including suffix.

    Returns
    -------

    w          : W
                 instance of spatial weights

    Examples
    --------
    >>> wr=rook_from_shapefile("../examples/columbus.shp")
    >>> wr.pct_nonzero
    0.083298625572678045

    Notes
    -----

    Rook contiguity defines as neighbors any pair of polygons that share a
    common edge in their polygon definitions.

    See Also
    --------

    pysal.weights.W
    """
    return buildContiguity(shapefile,criteria='rook')

def bishop_from_shapefile(shapefile):
    """once set operations on W are implemented do something like

    q=queen_from_shapefile(shapefile)
    r=rook_from_shapefile(shapefile)
    return q-r

    """

    raise NotImplementedError

if __name__ == "__main__":

    import doctest
    doctest.testmod()
