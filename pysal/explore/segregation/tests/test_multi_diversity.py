import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import MultiDiversity


class Multi_Diversity_Tester(unittest.TestCase):
    def test_Multi_Diversity(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiDiversity(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.9733112243997906)
        
        index_norm = MultiDiversity(df, groups_list, normalized = True)
        np.testing.assert_almost_equal(index_norm.statistic, 0.7020956383415715)


if __name__ == '__main__':
    unittest.main()