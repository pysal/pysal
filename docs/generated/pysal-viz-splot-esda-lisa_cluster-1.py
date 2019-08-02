# Imports

import matplotlib.pyplot as plt
from pysal.lib.weights.contiguity import Queen
from pysal.lib import examples
import geopandas as gpd
from pysal.explore.esda.moran import Moran_Local
from pysal.viz.splot.esda import lisa_cluster

# Data preparation and statistical analysis

link = examples.get_path('Guerry.shp')
gdf = gpd.read_file(link)
y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'
moran_loc = Moran_Local(y, w)

# Plotting

fig = lisa_cluster(moran_loc, gdf)
plt.show()
