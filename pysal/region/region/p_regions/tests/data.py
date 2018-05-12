"""
The data in this file describes the p-regions problem depicted in figure 1
on p. 106 in [DCM2011]_.
"""
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Polygon

from region.tests.util import region_list_from_array, convert_from_geodataframe
from region.util import dataframe_to_dict


attr = np.array([726.7, 623.6, 487.3,
                 200.4, 245.0, 481.0,
                 170.9, 225.9, 226.9])
attr_str = "attr"

gdf = GeoDataFrame(
        {attr_str: attr},
        geometry=[Polygon([(x, y),  # 3x3-grid
                           (x, y+1),
                           (x+1, y+1),
                           (x+1, y)]) for y in range(3) for x in range(3)]
)

optimal_clustering = region_list_from_array(np.array([0, 0, 0,
                                                      1, 1, 0,
                                                      1, 1, 1]))

# for tests with scalar attr & spatially_extensive_attr per area
attr = attr.reshape(-1, 1)
adj, graph, neighbors_dict, w = convert_from_geodataframe(gdf)
attr_dict = dataframe_to_dict(gdf, attr_str)

# for tests where attr & spatially_extensive_attr are vectors in each area
double_attr = np.column_stack((attr, attr))
double_attr_dict = dataframe_to_dict(gdf, [attr_str] * 2)
double_attr_str = [attr_str] * 2
