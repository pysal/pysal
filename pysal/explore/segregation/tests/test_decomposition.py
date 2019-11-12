import unittest
import pysal.lib
import geopandas as gpd
import numpy as np
from pysal.explore.segregation.aspatial import Dissim
from pysal.explore.segregation.decomposition import DecomposeSegregation


class Decomposition_Tester(unittest.TestCase):
    def test_Decomposition(self):
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        index1 = Dissim(s_map, 'HISP_', 'TOT_POP')
        index2 = Dissim(s_map, 'BLACK_', 'TOT_POP')
        res = DecomposeSegregation(index1, index2, counterfactual_approach = "composition")
        np.testing.assert_almost_equal(res.c_a, -0.16138819842911295)
        np.testing.assert_almost_equal(res.c_s, -0.005104643275796905)
        res.plot(plot_type = 'cdfs')
        res.plot(plot_type = 'maps')
        
        res = DecomposeSegregation(index1, index2, counterfactual_approach = "share")
        np.testing.assert_almost_equal(res.c_a, -0.1543828579279878)
        np.testing.assert_almost_equal(res.c_s, -0.012109983776922045)
        res.plot(plot_type = 'cdfs')
        res.plot(plot_type = 'maps')
        
        res = DecomposeSegregation(index1, index2, counterfactual_approach = "dual_composition")
        np.testing.assert_almost_equal(res.c_a, -0.16159526946235048)
        np.testing.assert_almost_equal(res.c_s, -0.004897572242559378)
        res.plot(plot_type = 'cdfs')
        res.plot(plot_type = 'maps')

if __name__ == '__main__':
    unittest.main()
