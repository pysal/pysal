import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import MultiSquaredCoefficientVariation


class Multi_Squared_Coefficient_of_Variation_Tester(unittest.TestCase):
    def test_Multi_Squared_Coefficient_of_Variation(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiSquaredCoefficientVariation(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.11875484641127525)


if __name__ == '__main__':
    unittest.main()