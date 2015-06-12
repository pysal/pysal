"""
Tests for statistics for gravity-style spatial interaction models
"""

__author__ = 'toshan'

import unittest
import numpy as np
import pandas as pd
import gravity as grav
import mle_stats as stats


class SingleParameter(unittest.TestCase):
    """Unit tests statistics when there is a single parameters estimated"""
    def setUp(self):
        self.f = np.array([0, 6469, 7629, 20036, 4690,
                           6194, 11688, 2243, 8857, 7248,
                           3559, 9221, 10099, 22866, 3388,
                           9986, 46618, 11639, 1380, 5261,
                           5985, 6731, 2704, 12250, 16132])
        self.o = np.repeat(1, 25)
        self.d = np.array(range(1, 26))
        self.dij = np.array([0, 576, 946, 597, 373,
                             559, 707, 1208, 602, 692,
                             681, 1934, 332, 595, 906,
                             425, 755, 672, 1587, 526,
                             484, 2141, 2182, 410, 540])
        self.pop = np.array([1596000, 2071000, 3376000, 6978000, 1345000,
                             2064000, 2378000, 1239000, 4435000, 1999000,
                             1274000, 7042000, 834000, 1268000, 1965000,
                             1046000, 12131000, 4824000, 969000, 2401000,
                             2410000, 2847000, 1425000, 1089000, 2909000])
        self.dt = pd.DataFrame({'origins': self.o,
                                'destinations': self.d,
                                'pop': self.pop,
                                'Dij': self.dij,
                                'flows': self.f})
    def test_single_parameter(self):
        model = grav.ProductionConstrained(self.dt, 'origins', 'destinations', 'flows',
            ['pop'], 'Dij', 'pow')
        ss = {'obs_mean_trip_len': 736.52834197296534,
              'pred_mean_trip_len': 734.40974204773784,
              'OD_pairs': 24,
              'predicted_flows': 242873.00000000003,
              'avg_dist_trav': 737.0,
              'num_destinations': 24,
              'observed_flows': 242873,
              'avg_dist': 851.0,
              'num_origins': 1}
        ps = {'beta': {'LL_zero_val': -3.057415839736517,
                       'relative_likelihood_stat': 24833.721614296166,
                       'standard_error': 0.0052734418614330883},
              'all_params': {'zero_vals_LL': -3.1780538303479453,
                             'mle_vals_LL': -3.0062909275101761},
              'pop': {'LL_zero_val': -3.1773474269437778,
                      'relative_likelihood_stat': 83090.010373874276,
                      'standard_error': 0.0027673052892085684}}
        fs = {'r_squared': 0.60516003720997413,
              'srmse': 0.57873206718148507}
        es = {'pred_obs_deviance': 0.1327,
              'entropy_ratio': 0.5642,
              'maximum_entropy': 3.1781,
              'max_pred_deviance': 0.1718,
              'variance_obs_entropy': 2.55421e-06,
              'predicted_entropy': 3.0063,
              't_stat_entropy': 66.7614,
              'max_obs_deviance': 0.3045,
              'observed_entropy': 2.8736,
              'variance_pred_entropy': 1.39664e-06}
        sys_stats = stats.sys_stats(model)
        self.assertAlmostEqual(model.system_stats['obs_mean_trip_len'], ss['obs_mean_trip_len'], 4)
        self.assertAlmostEqual(model.system_stats['pred_mean_trip_len'], ss['pred_mean_trip_len'], 4)
        self.assertAlmostEqual(model.system_stats['OD_pairs'], ss['OD_pairs'])
        self.assertAlmostEqual(model.system_stats['predicted_flows'], ss['predicted_flows'])
        self.assertAlmostEqual(model.system_stats['avg_dist_trav'], ss['avg_dist_trav'])
        self.assertAlmostEqual(model.system_stats['num_destinations'], ss['num_destinations'])
        self.assertAlmostEqual(model.system_stats['observed_flows'], ss['observed_flows'])
        self.assertAlmostEqual(model.system_stats['avg_dist'], ss['avg_dist'], 4)
        self.assertAlmostEqual(model.system_stats['num_origins'], ss['num_origins'])
        param_stats = stats.param_stats(model)
        self.assertAlmostEqual(model.parameter_stats['beta']['LL_zero_val'], ps['beta']['LL_zero_val'], 4)
        self.assertAlmostEqual(model.parameter_stats['beta']['relative_likelihood_stat'],
                                                  ps['beta']['relative_likelihood_stat'], 4)
        self.assertAlmostEqual(model.parameter_stats['beta']['standard_error'], ps['beta']['standard_error'], 4)
        self.assertAlmostEqual(model.parameter_stats['pop']['LL_zero_val'], ps['pop']['LL_zero_val'], 4)
        self.assertAlmostEqual(model.parameter_stats['pop']['relative_likelihood_stat'],
                                                  ps['pop']['relative_likelihood_stat'], 4)
        self.assertAlmostEqual(model.parameter_stats['pop']['standard_error'], ps['pop']['standard_error'], 4)
        self.assertAlmostEqual(model.parameter_stats['all_params']['zero_vals_LL'], ps['all_params']['zero_vals_LL'], 4)
        self.assertAlmostEqual(model.parameter_stats['all_params']['mle_vals_LL'], ps['all_params']['mle_vals_LL'], 4)
        fit_stats = stats.fit_stats(model)
        self.assertAlmostEqual(model.fit_stats['r_squared'], fs['r_squared'], 4)
        self.assertAlmostEqual(model.fit_stats['srmse'], fs['srmse'], 4)
        ent_stats = stats.ent_stats(model)
        self.assertAlmostEqual(model.entropy_stats['pred_obs_deviance'], es['pred_obs_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['entropy_ratio'], es['entropy_ratio'], 4)
        self.assertAlmostEqual(model.entropy_stats['maximum_entropy'], es['maximum_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['max_pred_deviance'], es['max_pred_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['variance_obs_entropy'], es['variance_obs_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['predicted_entropy'], es['predicted_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['t_stat_entropy'], es['t_stat_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['max_obs_deviance'], es['max_obs_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['observed_entropy'], es['observed_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['variance_pred_entropy'], es['variance_pred_entropy'], 4)

class MultipleParameter(unittest.TestCase):
    """Unit tests statistics when there are multiple parameters estimated"""
    def setUp(self):
        self.f = np.array([0, 180048, 79223, 26887, 198144, 17995, 35563, 30528, 110792,
                        283049, 0, 300345, 67280, 718673, 55094, 93434, 87987, 268458,
                        87267, 237229, 0, 281791, 551483, 230788, 178517, 172711, 394481,
                        29877, 60681, 286580, 0, 143860, 49892, 185618, 181868, 274629,
                        130830, 382565, 346407, 92308, 0, 252189, 192223, 89389, 279739,
                        21434, 53772, 287340, 49828, 316650, 0, 141679, 27409, 87938,
                        30287, 64645, 161645, 144980, 199466, 121366, 0, 134229, 289880,
                        21450, 43749, 97808, 113683, 89806, 25574, 158006, 0, 437255,
                        72114, 133122, 229764, 165405, 266305, 66324, 252039, 342948, 0])
        self.o = np.repeat(np.array(range(1, 10)), 9)
        self.d = np.tile(np.array(range(1, 10)), 9)
        self.dij = np.array([0, 219, 1009, 1514, 974, 1268, 1795, 2420, 3174,
                            219, 0, 831, 1336, 755, 1049, 1576, 2242, 2996,
                            1009, 831, 0, 505, 1019, 662, 933, 1451, 2205,
                            1514, 1336, 505, 0, 1370, 888, 654, 946, 1700,
                            974, 755, 1019, 1370, 0, 482, 1144, 2278, 2862,
                            1268, 1049, 662, 888, 482, 0, 662, 1795, 2380,
                            1795, 1576, 933, 654, 1144, 662, 0, 1287, 1779,
                            2420, 2242, 1451, 946, 2278, 1795, 1287, 0, 754,
                            3147, 2996, 2205, 1700, 2862, 2380, 1779, 754, 0])
        self.dt = pd.DataFrame({'Origin': self.o,
                                'Destination': self.d,
                                'flows': self.f,
                                'Dij': self.dij})
    def test_multiple_parameter(self):
        model = grav.DoublyConstrained(self.dt, 'Origin', 'Destination', 'flows', 'Dij', 'exp')
        ss = {'obs_mean_trip_len': 1250.9555521611339,
              'pred_mean_trip_len': 1250.9555521684863,
              'OD_pairs': 72, 'predicted_flows': 12314322.0,
              'avg_dist_trav': 1251.0, 'num_destinations': 9,
              'observed_flows': 12314322, 'avg_dist': 1414.0,
              'num_origins': 9}
        ps = {'beta': {'LL_zero_val': -4.1172103581711941,
                       'relative_likelihood_stat': 2053596.3814015209,
                       'standard_error': 4.9177433418433932e-07},
                        'all_params': {'zero_vals_LL': -4.1172102183395936,
                        'mle_vals_LL': -4.0338279201692675}}
        fs = {'r_squared': 0.89682406680906979,
              'srmse': 0.24804939821988789}
        es = {'pred_obs_deviance': 0.0314,
              'entropy_ratio': 0.8855,
              'maximum_entropy': 4.2767,
              'max_pred_deviance': 0.2429,
              'variance_obs_entropy': 3.667e-08,
              'predicted_entropy': 4.0338,
              't_stat_entropy': 117.1593,
              'max_obs_deviance': 0.2743,
              'observed_entropy': 4.0024,
              'variance_pred_entropy': 3.516e-08}
        sys_stats = stats.sys_stats(model)
        self.assertAlmostEqual(model.system_stats['obs_mean_trip_len'], ss['obs_mean_trip_len'], 4)
        self.assertAlmostEqual(model.system_stats['pred_mean_trip_len'], ss['pred_mean_trip_len'], 4)
        self.assertAlmostEqual(model.system_stats['OD_pairs'], ss['OD_pairs'])
        self.assertAlmostEqual(model.system_stats['predicted_flows'], ss['predicted_flows'])
        self.assertAlmostEqual(model.system_stats['avg_dist_trav'], ss['avg_dist_trav'])
        self.assertAlmostEqual(model.system_stats['num_destinations'], ss['num_destinations'])
        self.assertAlmostEqual(model.system_stats['observed_flows'], ss['observed_flows'])
        self.assertAlmostEqual(model.system_stats['avg_dist'], ss['avg_dist'], 4)
        self.assertAlmostEqual(model.system_stats['num_origins'], ss['num_origins'])
        param_stats = stats.param_stats(model)
        self.assertAlmostEqual(model.parameter_stats['beta']['LL_zero_val'], ps['beta']['LL_zero_val'], 4)
        self.assertAlmostEqual(model.parameter_stats['beta']['relative_likelihood_stat'],
                                                  ps['beta']['relative_likelihood_stat'], 4)
        self.assertAlmostEqual(model.parameter_stats['beta']['standard_error'], ps['beta']['standard_error'], 4)
        self.assertAlmostEqual(model.parameter_stats['all_params']['zero_vals_LL'], ps['all_params']['zero_vals_LL'], 4)
        self.assertAlmostEqual(model.parameter_stats['all_params']['mle_vals_LL'], ps['all_params']['mle_vals_LL'], 4)
        fit_stats = stats.fit_stats(model)
        self.assertAlmostEqual(model.fit_stats['r_squared'], fs['r_squared'], 4)
        self.assertAlmostEqual(model.fit_stats['srmse'], fs['srmse'], 4)
        ent_stats = stats.ent_stats(model)
        self.assertAlmostEqual(model.entropy_stats['pred_obs_deviance'], es['pred_obs_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['entropy_ratio'], es['entropy_ratio'], 4)
        self.assertAlmostEqual(model.entropy_stats['maximum_entropy'], es['maximum_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['max_pred_deviance'], es['max_pred_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['variance_obs_entropy'], es['variance_obs_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['predicted_entropy'], es['predicted_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['t_stat_entropy'], es['t_stat_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['max_obs_deviance'], es['max_obs_deviance'], 4)
        self.assertAlmostEqual(model.entropy_stats['observed_entropy'], es['observed_entropy'], 4)
        self.assertAlmostEqual(model.entropy_stats['variance_pred_entropy'], es['variance_pred_entropy'], 4)


if __name__ == '__main__':
    unittest.main()