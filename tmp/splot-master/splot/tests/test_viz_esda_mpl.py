import matplotlib.pyplot as plt
from libpysal.weights.contiguity import Queen
import libpysal as lp
from libpysal import examples
import geopandas as gpd
import numpy as np

from esda.moran import (Moran_Local, Moran, Moran_BV,
                        Moran_Local_BV, Moran_BV_matrix)
from splot.esda import (moran_scatterplot,
                        plot_moran_simulation,
                        plot_moran,
                        plot_moran_bv_simulation,
                        plot_moran_bv,
                        plot_local_autocorrelation,
                        lisa_cluster,
                        moran_facet)

from splot._viz_esda_mpl import (_moran_global_scatterplot,
                                 _moran_loc_scatterplot,
                                 _moran_bv_scatterplot,
                                 _moran_loc_bv_scatterplot)


def test_moran_scatterplot():
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['Suicids'].values
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calculate `esda.moran` Objects
    moran = Moran(y,w)
    moran_bv = Moran_BV(y, x, w)
    moran_loc = Moran_Local(y, w)
    moran_loc_bv = Moran_Local_BV(y, x, w)
    # try with p value so points are colored or warnings apply
    fig, _ = moran_scatterplot(moran, p=0.05)
    plt.close(fig)
    fig, _ = moran_scatterplot(moran_loc, p=0.05)
    plt.close(fig)
    fig, _ = moran_scatterplot(moran_bv, p=0.05)
    plt.close(fig)
    fig, _ = moran_scatterplot(moran_loc_bv, p=0.05)
    plt.close(fig)


def test_moran_global_scatterplot():
    # Load data and apply statistical analysis
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calc Global Moran
    w = Queen.from_dataframe(gdf)
    moran = Moran(y, w)
    # plot
    fig, _ = _moran_global_scatterplot(moran)
    plt.close(fig)
    # customize
    fig, _ = _moran_global_scatterplot(moran, zstandard=False,
                                       fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)


def test_plot_moran_simulation():
    # Load data and apply statistical analysis
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calc Global Moran
    w = Queen.from_dataframe(gdf)
    moran = Moran(y, w)
    # plot
    fig, _ = plot_moran_simulation(moran)
    plt.close(fig)
    # customize
    fig, _ = plot_moran_simulation(moran,
                                   fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)


def test_plot_moran():
    # Load data and apply statistical analysis
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calc Global Moran
    w = Queen.from_dataframe(gdf)
    moran = Moran(y, w)
    # plot
    fig, _ = plot_moran(moran)
    plt.close(fig)
    # customize
    fig, _ = plot_moran(moran, zstandard=False,
                        fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)

def test_moran_bv_scatterplot():
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['Suicids'].values
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calculate Bivariate Moran
    moran_bv = Moran_BV(x, y, w)
    # plot
    fig, _ = _moran_bv_scatterplot(moran_bv)
    plt.close(fig)
    # customize plot
    fig, _ = _moran_bv_scatterplot(moran_bv,
                                   fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)


def test_plot_moran_bv_simulation():
    # Load data and calculate weights
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['Suicids'].values
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calculate Bivariate Moran
    moran_bv = Moran_BV(x, y, w)
    # plot
    fig, _ = plot_moran_bv_simulation(moran_bv)
    plt.close(fig)
    # customize plot
    fig, _ = plot_moran_bv_simulation(moran_bv,
                                      fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)

def test_plot_moran_bv():
    # Load data and calculate weights
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['Suicids'].values
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calculate Bivariate Moran
    moran_bv = Moran_BV(x, y, w)
    # plot
    fig, _ = plot_moran_bv(moran_bv)
    plt.close(fig)
    # customize plot
    fig, _ = plot_moran_bv(moran_bv, fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)


def test_moran_loc_scatterplot():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = Moran_Local(y, w)

    # try with p value so points are colored
    fig, _ = _moran_loc_scatterplot(moran_loc, p=0.05)
    plt.close(fig)

    # try with p value and different figure size
    fig, _ = _moran_loc_scatterplot(moran_loc, p=0.05,
                                    fitline_kwds=dict(color='#4393c3'))
    plt.close(fig)


def test_lisa_cluster():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = Moran_Local(y, w)

    fig, _ = lisa_cluster(moran_loc, df)
    plt.close(fig)


def test_plot_local_autocorrelation():
    link = examples.get_path('columbus.shp')
    df = gpd.read_file(link)

    y = df['HOVAL'].values
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    moran_loc = Moran_Local(y, w)

    fig, _ = plot_local_autocorrelation(moran_loc, df, 'HOVAL', p=0.05)
    plt.close(fig)

    # also test with quadrant and mask
    fig, _ = plot_local_autocorrelation(moran_loc, df, 'HOVAL', p=0.05,
                                        region_column='POLYID',
                                        mask=['1', '2', '3'], quadrant=1)
    plt.close(fig)


def test_moran_loc_bv_scatterplot():
    link_to_data = examples.get_path('Guerry.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['Suicids'].values
    y = gdf['Donatns'].values
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Calculate Bivariate Moran
    moran_loc_bv = Moran_Local_BV(x, y, w)
    # try with p value so points are colored
    fig, _ = _moran_loc_bv_scatterplot(moran_loc_bv)
    plt.close(fig)

    # try with p value and different figure size
    fig, _ = _moran_loc_bv_scatterplot(moran_loc_bv, p=0.05)
    plt.close(fig)


def test_moran_facet():
    f = lp.io.open(examples.get_path("sids2.dbf"))
    varnames = ['SIDR74',  'SIDR79',  'NWR74',  'NWR79']
    vars = [np.array(f.by_col[var]) for var in varnames]
    w = lp.io.open(examples.get_path("sids2.gal")).read()
    # calculate moran matrix
    moran_matrix = Moran_BV_matrix(vars,  w,  varnames = varnames)
    # plot
    fig, axarr = moran_facet(moran_matrix)
    plt.close(fig)
    # customize
    fig, axarr = moran_facet(moran_matrix, scatter_glob_kwds=dict(color='r'),
                               fitline_bv_kwds=dict(color='y'))
    plt.close(fig)