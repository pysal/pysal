from pysal.lib.weights.contiguity import Queen
import pysal.lib
from pysal.lib import examples
import matplotlib.pyplot as plt
import geopandas as gpd

from pysal.viz.splot.libpysal import plot_spatial_weights

def test_plot_spatial_weights():
    # get data
    gdf = gpd.read_file(examples.get_path('43MUE250GC_SIR.shp'))
    gdf.head()
    # calculate weights
    weights = Queen.from_dataframe(gdf)
    # plot weights
    fig, _ = plot_spatial_weights(weights, gdf)
    plt.close(fig)
    # calculate nonplanar_joins
    wnp = pysal.lib.weights.util.nonplanar_neighbors(weights, gdf)
    # plot new joins
    fig2, _ = plot_spatial_weights(wnp, gdf)
    plt.close(fig2)
    #customize
    fig3, _ = plot_spatial_weights(wnp, gdf, nonplanar_edge_kws=dict(color='#4393c3'))
    plt.close(fig3)
    # plot in existing figure
    fig4, axs = plt.subplots(1,3)
    plot_spatial_weights(wnp, gdf, ax=axs[0])
    plt.close(fig4)

    # uses a column as the index for spatial weights object
    weights_index = Queen.from_dataframe(gdf, idVariable="CD_GEOCMU")
    fig, _ = plot_spatial_weights(weights_index, gdf, indexed_on="CD_GEOCMU")
    plt.close(fig)