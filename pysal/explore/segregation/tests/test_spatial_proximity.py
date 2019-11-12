import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial import SpatialProximity


class Spatial_Proximity_Tester(unittest.TestCase):
    def test_Spatial_Proximity(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = SpatialProximity(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 1.0026623464135092)


if __name__ == '__main__':
    unittest.main()
