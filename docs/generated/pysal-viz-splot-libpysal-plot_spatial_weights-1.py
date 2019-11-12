# Imports

from pysal.lib.weights.contiguity import Queen
import geopandas as gpd
import pysal.lib
from pysal.lib import examples
import matplotlib.pyplot as plt
from pysal.viz.splot.libpysal import plot_spatial_weights

# Data preparation and statistical analysis

gdf = gpd.read_file(examples.get_path('map_RS_BR.shp'))
weights = Queen.from_dataframe(gdf)
wnp = pysal.lib.weights.util.nonplanar_neighbors(weights, gdf)

# Plot weights

plot_spatial_weights(weights, gdf)
plt.show()

# Plot corrected weights

plot_spatial_weights(wnp, gdf)
plt.show()
