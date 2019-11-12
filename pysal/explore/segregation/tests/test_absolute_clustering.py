import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial import AbsoluteClustering


class Absolute_Clustering_Tester(unittest.TestCase):
    def test_Absolute_Clustering(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = AbsoluteClustering(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.005189287311955573)


if __name__ == '__main__':
    unittest.main()
