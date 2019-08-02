# Imports

from pysal.lib import examples
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pysal.viz.splot.mapping import vba_legend, mapclassify_bin

# Load Example Data

link_to_data = examples.get_path('columbus.shp')
gdf = gpd.read_file(link_to_data)
x = gdf['HOVAL'].values
y = gdf['CRIME'].values

# Classify your data

rgb_bins = mapclassify_bin(x, 'quantiles')
alpha_bins = mapclassify_bin(y, 'quantiles')

# Plot your legend

fig, _ = vba_legend(rgb_bins, alpha_bins, cmap='RdBu')
plt.show()
