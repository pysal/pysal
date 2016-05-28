from .weights import W
from pysal.cg import asShape
from pysal.core.FileIO import FileIO
from pysal.weights._contW_binning import ContiguityWeightsPolygons
from pysal.weights._contW_rtree import ContiguityWeights_rtree
from pysal.weights.util import get_ids
WT_TYPE = {'rook': 2, 'queen': 1}  # for _contW_Binning

class Rook(W):
    def __init__(self, polygons, **kwargs):
        """
        Construct a weights object from a collection of pysal polygons.

        Arguments
        ---------
        polygons    : iterable
        ids         :
        criterion   :
        driver      :
        **kw        : everything currently supported in W

        See Also
        ---------
        W
        """
        neighbors, ids = _build(polygons, criterion='rook', **kwargs)
        W.__init__(self, neighbors, ids=ids, **kwargs)
    
    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, full=False, **kwargs):
        """
        Rook contiguity weights from a polygon shapefile.

        Parameters
        ----------

        shapefile : string
                    name of polygon shapefile including suffix.
        sparse    : boolean
                    If True return WSP instance
                    If False return W instance

        Returns
        -------

        w          : W
                     instance of spatial weights

        Examples
        --------
        >>> wr=rook_from_shapefile(pysal.examples.get_path("columbus.shp"), "POLYID")
        >>> "%.3f"%wr.pct_nonzero
        '8.330'
        >>> wr=rook_from_shapefile(pysal.examples.get_path("columbus.shp"), sparse=True)
        >>> pct_sp = wr.sparse.nnz *1. / wr.n**2
        >>> "%.3f"%pct_sp
        '0.083'

        Notes
        -----

        Rook contiguity defines as neighbors any pair of polygons that share a
        common edge in their polygon definitions.

        See Also
        --------
        :class:`pysal.weights.W`
        """
        if idVariable is not None:
            ids = get_ids(filepath, idVariable) 
        else:
            ids = None
        w = cls(FileIO(filepath), ids=ids, **kwargs)
        w.set_shapefile(filepath, idVariable=dVariable, full=full)
    
    @classmethod
    def from_iterable(cls, iterable, **kwargs):
        """
        Construct a weights object from a collection of arbitrary polygons. This
        will cast the polygons to PySAL polygons, then build the W.

        Arguments
        ---------
        polygons    :
        ids         :
        criterion   :
        driver      :

        See Also
        ----------
        Queen
        """
        new_iterable = [asShape(shape) for shape in iterable]
        return cls(new_iterable, **kwargs)
    
    @classmethod
    def from_dataframe(cls, df, geomcol='geometry', **kwargs):
        """
        Construct a weights object from a pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Arguments
        ---------
        df
        geomcol

        See Also
        ---------
        Queen
        """
        ids = df.index.tolist()
        ids = df.index.tolist()
        return cls.from_iterable(df[geomcol].tolist(), ids=ids, **kwargs)

class Queen(W):
    def __init__(self, polygons, ids=None, criterion='queen', 
                 driver='binning', **kw):
        """
        Construct a weights object from a collection of pysal polygons.

        Arguments
        ---------
        polygons    : iterable
        ids         :
        criterion   :
        driver      :
        **kw        : everything currently supported in W

        See Also
        ---------
        W
        """
        neighbors, ids = _build(polygons, ids=ids, 
                                criterion=criterion, driver=driver)
        W.__init__(self, neighbors, ids=ids, **kw)
    
    @classmethod
    def from_shapefile(cls, filepath, idVariable=None, sparse=False, **kwargs):
        """
        Queen contiguity weights from a polygon shapefile.

        Parameters
        ----------

        shapefile   : string
                      name of polygon shapefile including suffix.
        idVariable  : string
                      name of a column in the shapefile's DBF to use for ids.
        sparse    : boolean
                    If True return WSP instance
                    If False return W instance
        Returns
        -------

        w            : W
                       instance of spatial weights

        Examples
        --------
        >>> wq=Queen.from_shapefile(pysal.examples.get_path("columbus.shp"))
        >>> "%.3f"%wq.pct_nonzero
        '9.829'
        >>> wq=Queen.from_shapefile(pysal.examples.get_path("columbus.shp"),"POLYID")
        >>> "%.3f"%wq.pct_nonzero
        '9.829'
        >>> wq=Queen.from_shapefile(pysal.examples.get_path("columbus.shp"), sparse=True)
        >>> pct_sp = wq.sparse.nnz *1. / wq.n**2
        >>> "%.3f"%pct_sp
        '0.098'

        Notes

        Queen contiguity defines as neighbors any pair of polygons that share at
        least one vertex in their polygon definitions.

        See Also
        --------
        :class:`pysal.weights.W`

        """
        if idVariable is not None:
            ids = get_ids(filepath, idVariable) 
        else:
            ids = None
        iterable = FileIO(filepath)
        return cls.from_iterable(iterable, sparse=sparse, **kwargs)

    @classmethod
    def from_iterable(cls, iterable, sparse=True, **kwargs):
        """
        Construct a weights object from a collection of arbitrary polygons. This
        will cast the polygons to PySAL polygons, then build the W.

        Arguments
        ---------
        polygons    :
        ids         :
        criterion   :
        driver      :

        See Also
        ----------
        Queen
        """
        new_iterable = [asShape(shape) for shape in iterable]
        
        w = cls(new_iterable, **kwargs) 
        if sparse:
            w = WSP.from_W(w)
        
        return w

    @classmethod
    def from_dataframe(cls, df, geomcol='geometry', **kwargs):
        """
        Construct a weights object from a pandas dataframe with a geometry
        column. This will cast the polygons to PySAL polygons, then build the W
        using ids from the dataframe.

        Arguments
        ---------
        df
        geomcol

        See Also
        ---------
        Queen
        """
        ids = df.index.tolist()
        return cls.from_iterable(df[geomcol].tolist(), ids=ids, **kwargs)

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
