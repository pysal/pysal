# Imports

import matplotlib.pyplot as plt
from pysal.lib.weights.contiguity import Queen
from pysal.lib import examples
import geopandas as gpd
from pysal.explore.esda.moran import Moran
from pysal.viz.splot.esda import plot_moran_simulation

# Load data and calculate weights

link_to_data = examples.get_path('Guerry.shp')
gdf = gpd.read_file(link_to_data)
y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'

# Calculate Global Moran

moran = Moran(y, w)

# plot

plot_moran_simulation(moran)
plt.show()

# customize plot

plot_moran_simulation(moran, fitline_kwds=dict(color='#4393c3'))
plt.show()
