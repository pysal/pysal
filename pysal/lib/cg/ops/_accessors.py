import functools as _f

__all__ = [ 'area', 'bbox', 'bounding_box', 'centroid', 'holes', 'len', 
            'parts', 'perimeter', 'segments', 'vertices']

def get_attr(df, geom_col='geometry', inplace=False, attr=None):
    outval = df[geom_col].apply(lambda x: x.__getattribute__(attr))
    if inplace:
        outcol = 'shape_{}'.format(func.__name__)
        df[outcol] = outval
        return None
    return outval

_doc_template =\
""" 
Tabular accessor to grab a geometric object's {n} attribute

Arguments
---------
df      :   pandas.DataFrame
            a pandas dataframe with a geometry column
geom_col:   string
            the name of the column in df containing the geometry
inplace :   bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived column will be under 'shape_{n}'

Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a series. 

See Also
---------
For further documentation about the attributes of the object in question, refer
to shape classes in pysal.cg.shapes
"""

_accessors = dict()
for k in __all__:
    _accessors[k] = _f.partial(get_attr, attr=k)
    _accessors[k].__doc__ = _doc_template.format(n=k)

globals().update(_accessors)
