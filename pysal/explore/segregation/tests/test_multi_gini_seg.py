import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import MultiGiniSeg


class Multi_Gini_Seg_Tester(unittest.TestCase):
    def test_Multi_Gini_Seg(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiGiniSeg(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.5456349992598081)


if __name__ == '__main__':
    unittest.main()