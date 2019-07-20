import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import SimpsonsConcentration


class Simpsons_Concentration_Tester(unittest.TestCase):
    def test_Simpsons_Concentration(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
        df = s_map[groups_list]
        index = SimpsonsConcentration(df, groups_list)
        np.testing.assert_almost_equal(index.statistic, 0.49182413151957904)


if __name__ == '__main__':
    unittest.main()