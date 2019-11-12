import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import DensityCorrectedDissim


class Density_Corrected_Dissim_Tester(unittest.TestCase):

    def test_Density_Corrected_Dissim(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = DensityCorrectedDissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.295205155464069)

if __name__ == '__main__':
    unittest.main()
