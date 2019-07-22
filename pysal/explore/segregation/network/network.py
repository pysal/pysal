"""Calculate street network-based segregation measures."""

__author__ = "Elijah Knaap <elijah.knaap@ucr.edu> Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
from warnings import warn
from pysal.explore.segregation.util import project_gdf
import os
import sys
try:
    import pandana as pdna
    from urbanaccess.osm.load import ua_network_from_bbox
except ImportError:
    warn(
        "You need pandana and urbanaccess to work with segregation's network module\n"
        "You can install them with  `pip install urbanaccess pandana` "
        "or `conda install -c udst pandana urbanaccess`")


# This class allows us to hide the diagnostic messages from urbanaccess if the `quiet` flag is set
class _HiddenPrints:  # from https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_osm_network(geodataframe, maxdist=5000, quiet=True, **kwargs):
    """Download a street network from OSM.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geopandas.GeoDataFrame of the study area.
        Coordinate system should be in WGS84
    maxdist : int
        Total distance (in meters) of the network queries you may need.
        This is used to buffer the network to ensure theres enough to satisfy
        your largest query, otherwise there may be edge effects.
    quiet: bool
        If True, diagnostic messages from urbanaccess will be suppressed
    **kwargs : dict
        additional kwargs passed through to
        urbanaccess.ua_network_from_bbox

    Returns
    -------
    pandana.Network
        A pandana Network instance for use in accessibility calculations or
        spatial segregation measures that include a distance decay

    """
    gdf = geodataframe.copy()
    gdf = project_gdf(gdf)
    gdf = gdf.buffer(maxdist)
    bounds = gdf.to_crs(epsg=4326).total_bounds

    if quiet:
        print('Downloading data from OSM. This may take awhile.')
        with _HiddenPrints():
            net = ua_network_from_bbox(bounds[1], bounds[0], bounds[3],
                                       bounds[2], **kwargs)
    else:
        net = ua_network_from_bbox(bounds[1], bounds[0], bounds[3], bounds[2],
                                   **kwargs)
    print("Building network")
    network = pdna.Network(net[0]["x"], net[0]["y"], net[1]["from"],
                           net[1]["to"], net[1][["distance"]])

    return network


def calc_access(geodataframe,
                network,
                distance=2000,
                decay="linear",
                variables=None,
                precompute=True):
    """Calculate access to population groups.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geodataframe with demographic data
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_osm_network`
        or via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".
    variables : list
        list of variable names present on gdf that should be calculated
    precompute: bool (default True)
        whether pandana should precompute the distance matrix. It can only be
        precomputed once, so If you plan to pass the same network to this
        function several times, you should set precompute=False for later runs

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns, `total_population` and `group_population`
        which represent the total number of each group that can be reached
        within the supplied `distance` parameter. The DataFrame is indexed
        on node_ids

    """
    if precompute:
        network.precompute(distance)

    geodataframe["node_ids"] = network.get_node_ids(geodataframe.centroid.x,
                                                    geodataframe.centroid.y)

    access = []
    for variable in variables:
        network.set(geodataframe.node_ids,
                    variable=geodataframe[variable],
                    name=variable)

        access_pop = network.aggregate(distance,
                                       type="sum",
                                       decay=decay,
                                       name=variable)

        access.append(access_pop)
    names = ["acc_" + variable for variable in variables]
    access = pd.DataFrame(dict(zip(names, access)))

    return access
