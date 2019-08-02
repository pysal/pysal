# Imports

import matplotlib.pyplot as plt
from pysal.lib.weights.contiguity import Queen
from pysal.lib import examples
import geopandas as gpd
from pysal.explore.esda.moran import Moran_BV
from pysal.viz.splot.esda import plot_moran_bv

# Load data and calculate weights

link_to_data = examples.get_path('Guerry.shp')
gdf = gpd.read_file(link_to_data)
x = gdf['Suicids'].values
y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'

# Calculate Bivariate Moran

moran_bv = Moran_BV(x, y, w)

# plot

plot_moran_bv(moran_bv)
plt.show()

# customize plot

plot_moran_bv(moran_bv, fitline_kwds=dict(color='#4393c3'))
plt.show()
