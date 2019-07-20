import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import MultiRelativeDiversity


class Multi_Relative_Diversity_Tester(unittest.TestCase):
    def test_Multi_Relative_Diversity(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiRelativeDiversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.15820019878220337)


if __name__ == '__main__':
    unittest.main()