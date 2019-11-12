import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import ModifiedDissim


class Modified_Dissim_Tester(unittest.TestCase):
    def test_Modified_Dissim(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        np.random.seed(1234)
        index = ModifiedDissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.31075891224250635)


if __name__ == '__main__':
    unittest.main()
