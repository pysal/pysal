# Imports

import matplotlib.pyplot as plt
import pysal.lib as lp
import numpy as np
import geopandas as gpd
from pysal.explore.esda.moran import Moran_BV_matrix
from pysal.viz.splot.esda import moran_facet

# Load data and calculate Moran Local statistics

f = gpd.read_file(lp.examples.get_path("sids2.dbf"))
varnames = ['SIDR74',  'SIDR79',  'NWR74',  'NWR79']
vars = [np.array(f[var]) for var in varnames]
w = lp.io.open(lp.examples.get_path("sids2.gal")).read()
moran_matrix = Moran_BV_matrix(vars,  w,  varnames = varnames)

# Plot

fig, axarr = moran_facet(moran_matrix)
plt.show()

# Customize plot

fig, axarr = moran_facet(moran_matrix,
                         fitline_bv_kwds=dict(color='#4393c3'))
plt.show()
