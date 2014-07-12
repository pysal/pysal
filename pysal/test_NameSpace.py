import os
import unittest
import pysal


class TestNameSpace(unittest.TestCase):
    """
        This test makes sure we don't remove anything from the pysal NameSpace that
        1.0 users might expect to be there.  1.0 Namespace was taken from the 1.1
        Code sprint wave, with special names removes (__all__, etc)
    """
    def test_contents(self):
        namespace_v1_0 = ['Box_Plot', 'DistanceBand', 'Equal_Interval',
                          'Fisher_Jenks', 'Geary', 'Jenks_Caspall',
                          'Jenks_Caspall_Forced', 'Jenks_Caspall_Sampled',
                          'Join_Counts', 'K_classifiers', 'Kernel',
                          'LISA_Markov', 'Markov', 'Max_P_Classifier',
                          'Maximum_Breaks', 'Maxp', 'Maxp_LISA', 'Moran',
                          'Moran_BV', 'Moran_BV_matrix', 'Moran_Local',
                          'Natural_Breaks', 'Percentiles', 'Quantiles',
                          'SpatialTau', 'Spatial_Markov', 'Std_Mean', 'Theil',
                          'TheilD', 'TheilDSim', 'Theta', 'User_Defined', 'W', 'adaptive_kernelW',
                          'adaptive_kernelW_from_shapefile', 'bin', 'bin1d',
                          'binC', 'buildContiguity', 'cg', 'comb', 'common',
                          'core', 'directional', 'ergodic', 'esda', 'full',
                          'gadf', 'higher_order', 'inequality', 'kernelW',
                          'kernelW_from_shapefile', 'knnW', 'knnW_from_array',
                          'knnW_from_shapefile', 'lag_spatial', 'lat2W',
                          'min_threshold_dist_from_shapefile', 'open',
                          'order', 'quantile', 'queen_from_shapefile',
                          'block_weights', 'region', 'remap_ids',
                          'rook_from_shapefile', 'shimbel', 'spatial_dynamics',
                          'threshold_binaryW_from_array', 'threshold_binaryW_from_shapefile',
                          'threshold_continuousW_from_array', 'threshold_continuousW_from_shapefile',
                          'version', 'w_difference', 'w_intersection', 'w_subset',
                          'w_symmetric_difference', 'w_union', 'weights']

        current_namespace = dir(pysal)
        for item in namespace_v1_0:
            self.assertTrue(item in current_namespace)
        for item in current_namespace:
            if item not in namespace_v1_0 and not item.startswith('__'):
                print item, "added to name space"


suite = unittest.TestLoader().loadTestsFromTestCase(TestNameSpace)

if __name__ == '__main__':
    unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(suite)
