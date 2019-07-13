import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial_indexes import Spatial_Exposure


class Spatial_Exposure_Tester(unittest.TestCase):

    def test_Spatial_Exposure(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Spatial_Exposure(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.8396583368412371)

if __name__ == '__main__':
    unittest.main()
