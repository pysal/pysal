import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.local import MultiLocalEntropy


class Multi_Local_Entropy_Tester(unittest.TestCase):
    def test_Multi_Local_Entropy(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiLocalEntropy(df, groups_list)
        np.testing.assert_almost_equal(index.statistics[0:10], np.array([0.24765538, 0.40474253, 0.50900607, 0.21433739, 0.16148146,
																		 0.21454691, 0.08889013, 0.08132889, 0.06782401, 0.18127186]))


if __name__ == '__main__':
    unittest.main()
