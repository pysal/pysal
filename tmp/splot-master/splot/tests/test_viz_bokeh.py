from libpysal.weights.contiguity import Queen
from libpysal import examples
import geopandas as gpd
import esda

from splot.bk import (plot_choropleth, lisa_cluster,
                      moran_scatterplot, plot_local_autocorrelation)


def test_plot_choropleth():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    w = Queen.from_dataframe(df)
    w.transform = 'r'

    TOOLS = "tap,help"
    fig = plot_choropleth(df, 'HOVAL', title='columbus',
                          reverse_colors=True, tools=TOOLS)


def test_lisa_cluster():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = esda.moran.Moran_Local(y, w)

    TOOLS = "tap,reset,help"
    fig = lisa_cluster(moran_loc, df, p=0.05, tools=TOOLS)


def test_moran_scatterplot():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = esda.moran.Moran_Local(y, w)

    fig = moran_scatterplot(moran_loc, p=0.05)


def test_plot_local_autocorrelation():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = esda.moran.Moran_Local(y, w)

    fig = plot_local_autocorrelation(moran_loc, df, 'HOVAL')
