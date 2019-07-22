import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.local import LocalRelativeCentralization


class Local_Relative_Centralization_Tester(unittest.TestCase):
    def test_Local_Relative_Centralization(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'BLACK_', 'TOT_POP']]
        index = LocalRelativeCentralization(df, 'BLACK_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistics[0:10], np.array([0.03443055, -0.29063264, -0.19110976,  0.24978919,  0.01252249,
																		 0.61152941,  0.78917647,  0.53129412,  0.04436346, -0.20216325]))


if __name__ == '__main__':
    unittest.main()
