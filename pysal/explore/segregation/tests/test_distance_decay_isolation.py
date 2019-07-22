import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial import DistanceDecayIsolation


class Distance_Decay_Isolation_Tester(unittest.TestCase):
    def test_Distance_Decay_Isolation(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = DistanceDecayIsolation(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.1562162475606278)


if __name__ == '__main__':
    unittest.main()
