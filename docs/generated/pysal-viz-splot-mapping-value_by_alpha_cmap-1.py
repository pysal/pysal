# Imports

from pysal.lib import examples
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pysal.viz.splot.mapping import value_by_alpha_cmap

# Load Example Data

link_to_data = examples.get_path('columbus.shp')
gdf = gpd.read_file(link_to_data)
x = gdf['HOVAL'].values
y = gdf['CRIME'].values

# Create rgba values

rgba, _ = value_by_alpha_cmap(x, y)

# Create divergent rgba and change Colormap

div_rgba, _ = value_by_alpha_cmap(x, y, cmap='seismic', divergent=True)

# Create rgba values with reverted alpha values

rev_rgba, _  = value_by_alpha_cmap(x, y, cmap='RdBu', revert_alpha=True)
