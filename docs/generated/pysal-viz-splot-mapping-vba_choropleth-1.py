# Imports

from pysal.lib import examples
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pysal.viz.splot.mapping import vba_choropleth

# Load Example Data

link_to_data = examples.get_path('columbus.shp')
gdf = gpd.read_file(link_to_data)

# Plot a Value-by-Alpha map

fig, _ = vba_choropleth('HOVAL', 'CRIME', gdf)
plt.show()

# Plot a Value-by-Alpha map with reverted alpha values

fig, _ = vba_choropleth('HOVAL', 'CRIME', gdf, cmap='RdBu',
                        revert_alpha=True)
plt.show()

# Plot a Value-by-Alpha map with classified alpha and rgb values

fig, axs = plt.subplots(2,2, figsize=(20,10))
vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,0],
               rgb_mapclassify=dict(classifier='quantiles', k=3), 
               alpha_mapclassify=dict(classifier='quantiles', k=3))
vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[0,1],
               rgb_mapclassify=dict(classifier='natural_breaks'), 
               alpha_mapclassify=dict(classifier='natural_breaks'))
vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,0],
               rgb_mapclassify=dict(classifier='std_mean'), 
               alpha_mapclassify=dict(classifier='std_mean'))
vba_choropleth('HOVAL', 'CRIME', gdf, cmap='viridis', ax = axs[1,1],
               rgb_mapclassify=dict(classifier='fisher_jenks', k=3), 
               alpha_mapclassify=dict(classifier='fisher_jenks', k=3))
plt.show()

# Pass in a list of colors instead of a cmap

color_list = ['#a1dab4','#41b6c4','#225ea8']
vba_choropleth('HOVAL', 'CRIME', gdf, cmap=color_list,
               rgb_mapclassify=dict(classifier='quantiles', k=3), 
               alpha_mapclassify=dict(classifier='quantiles'))
plt.show()

# Add a legend and use divergent alpha values

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
vba_choropleth('HOVAL', 'CRIME', gdf, divergent=True,
               alpha_mapclassify=dict(classifier='quantiles', k=5),
               rgb_mapclassify=dict(classifier='quantiles', k=5),
               legend=True, ax=ax)
plt.show()
