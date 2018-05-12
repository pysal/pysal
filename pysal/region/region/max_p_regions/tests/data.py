"""
The data in this file describes the max-p-regions problem depicted in figure 2
on p. 402 in [DAR2012]_.
"""
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from region.tests.util import region_list_from_array, convert_from_geodataframe
from region.util import dataframe_to_dict

attr = np.array([350.2, 400.5, 430.8,
                 490.4, 410.9, 450.4,
                 560.1, 500.7, 498.6])
spatially_extensive_attr = np.array([30, 25, 31,
                                     28, 32, 30,
                                     35, 27, 33])
threshold = 120
optimal_clustering = region_list_from_array(np.array([0, 0, 0,
                                                      1, 0, 0,
                                                      1, 1, 1]))

attr_str = "attr"
spatially_extensive_attr_str = "spatially_extensive_attr"
gdf = GeoDataFrame(
        {attr_str: attr,
         spatially_extensive_attr_str: spatially_extensive_attr},
        geometry=[Polygon([(x, y),
                           (x, y+1),
                           (x+1, y+1),
                           (x+1, y)]) for y in range(3) for x in range(3)]
)

# for tests with scalar attr & spatially_extensive_attr per area
attr = attr.reshape(-1, 1)
spatially_extensive_attr = spatially_extensive_attr.reshape(-1, 1)

adj, graph, neighbors_dict, w = convert_from_geodataframe(gdf)
attr_dict = dataframe_to_dict(gdf, attr_str)
spatially_extensive_attr_dict = dataframe_to_dict(gdf,
                                                  spatially_extensive_attr_str)
# for tests where attr & spatially_extensive_attr are vectors in each area
double_attr = np.column_stack((attr, attr))
double_spatially_extensive_attr = np.column_stack((spatially_extensive_attr,
                                                   spatially_extensive_attr))
double_threshold = np.hstack((threshold, threshold))
double_attr_dict = dataframe_to_dict(gdf, [attr_str] * 2)
double_spatially_extensive_attr_dict = dataframe_to_dict(
        gdf, [spatially_extensive_attr_str] * 2)
