import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial_indexes import Absolute_Centralization


class Absolute_Centralization_Tester(unittest.TestCase):

    def test_Absolute_Centralization(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Absolute_Centralization(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.6891422368736286)

if __name__ == '__main__':
    unittest.main()
