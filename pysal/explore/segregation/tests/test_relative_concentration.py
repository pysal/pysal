import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial import RelativeConcentration


class Relative_Concentration_Tester(unittest.TestCase):
    def test_Relative_Concentration(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = RelativeConcentration(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.12733820870675222)


if __name__ == '__main__':
    unittest.main()
