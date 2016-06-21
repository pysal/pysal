_univ_doc_template =\
""" 
Tabular accessor to compute a {n} statistic

Arguments
---------
df      :   pandas.DataFrame
            a pandas dataframe with a geometry column
cols    :   string or list of string
            string or list of strings denoting which columns on which to compute
            the statistic
w       :   pysal weights object
            a weights object aligned with the dataframe. If not provided, this
            is searched for in the dataframe's metadata
inplace :   bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named 'column_{nl}'
pvalue  :   string
            a string denoting which pvalue should be returned. Refer to the
            the {n} statistic's documentation for available p-values
outvals :   list of strings
            list of arbitrary attributes to return as columns from the {n} statistic

Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a copy of the dataframe with the relevant columns attached.

See Also
---------
For further documentation, refer to {n}
"""

_bv_doc_template =\
""" 
Tabular accessor to compute a {n} statistic

Arguments
---------
df      :   pandas.DataFrame
            a pandas dataframe with a geometry column
cols    :   string or list of string
            string or list of strings denoting which columns on which to compute
            the statistic
w       :   pysal weights object
            a weights object aligned with the dataframe. If not provided, this
            is searched for in the dataframe's metadata
inplace :   bool
            a boolean denoting whether to operate on the dataframe inplace or to
            return a series contaning the results of the computation. If
            operating inplace, the derived columns will be named 'column_{nl}'
pvalue  :   string
            a string denoting which pvalue should be returned. Refer to the
            the {n} statistic's documentation for available p-values
outvals :   list of strings
            list of arbitrary attributes to return as columns from the {n} statistic

Returns
--------
If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
returns a copy of the dataframe with the relevant columns attached.

See Also
---------
For further documentation, refer to {n}
"""

_accessors_template =\
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

