import pysal as ps
from pysal.spreg import handler as h
from pysal.contrib.pdutilities.dbf_utilities import dbf2df

data = ps.open(ps.examples.get_path('columbus.dbf'))
df = dbf2df(ps.examples.get_path('columbus.dbf'))

y = data.by_col_array(['HOVAL'])
X = data.by_col_array(['INC', 'CRIME'])

refreg = ps.spreg.OLS(y, X)
