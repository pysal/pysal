# Imports

import matplotlib.pyplot as plt
from pysal.lib.weights.contiguity import Queen
from pysal.lib import examples
import geopandas as gpd
from pysal.explore.esda.moran import Moran_Local
from pysal.viz.splot.esda import plot_local_autocorrelation

# Data preparation and analysis

link = examples.get_path('Guerry.shp')
gdf = gpd.read_file(link)
y = gdf['Donatns'].values
w = Queen.from_dataframe(gdf)
w.transform = 'r'
moran_loc = Moran_Local(y, w)

# Plotting with quadrant mask and region mask

fig = plot_local_autocorrelation(moran_loc, gdf, 'Donatns', p=0.05,
                                 region_column='Dprtmnt',
                                 mask=['Ain'], quadrant=1)
plt.show()
