import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.local import MultiLocalSimpsonConcentration


class Multi_Local_Simpson_Concentration_Tester(unittest.TestCase):
    def test_Multi_Local_Simpson_Concentration(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiLocalSimpsonConcentration(df, groups_list)
        np.testing.assert_almost_equal(index.statistics[0:10], np.array([0.84564007, 0.66608405, 0.50090253, 0.8700551 , 0.90194944,
																		 0.86871822, 0.95552644, 0.9601067 , 0.96276946, 0.88241452]))


if __name__ == '__main__':
    unittest.main()
