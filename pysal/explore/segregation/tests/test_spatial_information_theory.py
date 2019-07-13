import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial_indexes import Spatial_Information_Theory


class Spatial_Information_Theory_Tester(unittest.TestCase):

    def test_Spatial_Information_Theory(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Spatial_Information_Theory(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.778177518074913)

if __name__ == '__main__':
    unittest.main()
