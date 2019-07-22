import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.local import MultiLocalSimpsonInteraction


class Multi_Local_Simpson_Interaction_Tester(unittest.TestCase):
    def test_Multi_Local_Simpson_Interaction(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiLocalSimpsonInteraction(df, groups_list)
        np.testing.assert_almost_equal(index.statistics[0:10], np.array([0.15435993, 0.33391595, 0.49909747, 0.1299449 , 0.09805056,
																		 0.13128178, 0.04447356, 0.0398933 , 0.03723054, 0.11758548]))


if __name__ == '__main__':
    unittest.main()
