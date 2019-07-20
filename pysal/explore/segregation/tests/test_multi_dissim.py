import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import MultiDissim


class Multi_Dissim_Tester(unittest.TestCase):
    def test_Multi_Dissim(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiDissim(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.41340872573177806)


if __name__ == '__main__':
    unittest.main()