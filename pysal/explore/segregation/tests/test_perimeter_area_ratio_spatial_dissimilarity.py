import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.spatial import PerimeterAreaRatioSpatialDissim


class Perimeter_Area_Ratio_Spatial_Dissim_Tester(unittest.TestCase):
    def test_Perimeter_Area_Ratio_Spatial_Dissim(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = PerimeterAreaRatioSpatialDissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.3111718061947464)


if __name__ == '__main__':
    unittest.main()
