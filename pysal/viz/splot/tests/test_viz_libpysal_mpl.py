import pysal.lib.api as lp
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
    weights = lp.Queen.from_dataframe(gdf)
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