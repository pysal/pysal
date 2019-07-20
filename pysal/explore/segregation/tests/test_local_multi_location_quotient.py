import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.local import MultiLocationQuotient


class Multi_Location_Quotient_Tester(unittest.TestCase):
    def test_Multi_Location_Quotient(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = MultiLocationQuotient(df, groups_list)
        np.testing.assert_almost_equal(index.statistics[0:3,0:3], np.array([[1.36543221, 0.07478049, 0.16245651],
																			[1.18002164, 0.        , 0.14836683],
																			[0.68072696, 0.03534425, 0.        ]]))


if __name__ == '__main__':
    unittest.main()
