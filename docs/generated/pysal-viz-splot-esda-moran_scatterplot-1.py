# Imports

import matplotlib.pyplot as plt
from pysal.lib.weights.contiguity import Queen
from pysal.lib import examples
import geopandas as gpd
from pysal.explore.esda.moran import (Moran, Moran_BV,
                        Moran_Local, Moran_Local_BV)
from pysal.viz.splot.esda import moran_scatterplot

# Load data and calculate weights

link_to_data = examples.get_path('Guerry.shp')
gdf = gpd.read_file(link_to_data)
x = gdf['Suicids'].values
y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'

# Calculate esda.moran Objects

moran = Moran(y, w)
moran_bv = Moran_BV(y, x, w)
moran_loc = Moran_Local(y, w)
moran_loc_bv = Moran_Local_BV(y, x, w)

# Plot

fig, axs = plt.subplots(2, 2, figsize=(10,10),
                        subplot_kw={'aspect': 'equal'})
moran_scatterplot(moran, p=0.05, ax=axs[0,0])
moran_scatterplot(moran_loc, p=0.05, ax=axs[1,0])
moran_scatterplot(moran_bv, p=0.05, ax=axs[0,1])
moran_scatterplot(moran_loc_bv, p=0.05, ax=axs[1,1])
plt.show()
