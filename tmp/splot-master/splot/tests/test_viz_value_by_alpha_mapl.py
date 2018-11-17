import libpysal as lp
from libpysal import examples
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from splot.mapping import (value_by_alpha_cmap,
                           vba_choropleth,
                           mapclassify_bin,
                           vba_legend)


def test_value_by_alpha_cmap():
    # data
    link_to_data = examples.get_path('columbus.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['HOVAL'].values
    y = gdf['CRIME'].values
    # create cmap
    rgba, cmap = value_by_alpha_cmap(x, y)
    # create divergent rgba
    div_rgba, _ = value_by_alpha_cmap(x, y, cmap='seismic', divergent=True)
    # create reverted rgba
    rev_rgba, _ = value_by_alpha_cmap(x, y, cmap='RdBu', revert_alpha=True)


def test_vba_choropleth():
    # data
    link_to_data = examples.get_path('columbus.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['HOVAL'].values
    y = gdf['CRIME'].values
    # plot
    fig, _ = vba_choropleth(x, y, gdf)
    plt.close(fig)
    # plot with divergent and reverted alpha
    fig, _ = vba_choropleth(x, y, gdf, cmap='RdBu',
                            divergent=True,
                            revert_alpha=True)
    plt.close(fig)
    # plot with classified alpha and rgb
    fig, _ = vba_choropleth(x, y, gdf, cmap='RdBu',
                            alpha_mapclassify=dict(classifier='quantiles'),
                            rgb_mapclassify=dict(classifier='quantiles'))
    plt.close(fig)
    # plot classified with legend
    fig, _ = vba_choropleth(x, y, gdf,
                            alpha_mapclassify=dict(classifier='std_mean'),
                            rgb_mapclassify=dict(classifier='std_mean'),
                            legend=True)
    plt.close(fig)


def test_vba_legend():
    # data
    link_to_data = examples.get_path('columbus.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['HOVAL'].values
    y = gdf['CRIME'].values
    # classify data
    rgb_bins = mapclassify_bin(x, 'quantiles')
    alpha_bins = mapclassify_bin(y, 'quantiles')
    # plot legend
    fig, _ = vba_legend(rgb_bins, alpha_bins, cmap='RdBu')
    plt.close(fig)


def test_mapclassify_bin():
    # data
    link_to_data = examples.get_path('columbus.shp')
    gdf = gpd.read_file(link_to_data)
    x = gdf['HOVAL'].values
    # quantiles
    mapclassify_bin(x, 'quantiles')
    mapclassify_bin(x, 'quantiles', k=3)
    # box_plot
    mapclassify_bin(x, 'box_plot')
    mapclassify_bin(x, 'box_plot', hinge=2)
    # headtail_breaks
    mapclassify_bin(x, 'headtail_breaks')   
    # percentiles
    mapclassify_bin(x, 'percentiles')
    mapclassify_bin(x, 'percentiles', pct=[25,50,75,100])
    # std_mean
    mapclassify_bin(x, 'std_mean')
    mapclassify_bin(x, 'std_mean', multiples=[-1,-0.5,0.5,1])
    # maximum_breaks
    mapclassify_bin(x, 'maximum_breaks')
    mapclassify_bin(x, 'maximum_breaks', k=3, mindiff=0.1)
    # natural_breaks, max_p_classifier
    mapclassify_bin(x, 'natural_breaks')
    mapclassify_bin(x, 'max_p_classifier', k=3, initial=50)
    # user_defined
    mapclassify_bin(x, 'user_defined', bins=[20, max(x)])
