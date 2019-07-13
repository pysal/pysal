import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.non_spatial_indexes import Bias_Corrected_Dissim


class Bias_Corrected_Dissim_Tester(unittest.TestCase):

    def test_Bias_Corrected_Dissim(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        np.random.seed(1234)
        index = Bias_Corrected_Dissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.32136474449360836)

if __name__ == '__main__':
    unittest.main()
