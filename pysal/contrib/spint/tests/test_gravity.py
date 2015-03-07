"""
Tests for gravity-style spatial interaction models
"""

__author__ = 'toshan'

import unittest
import numpy as np
import pandas as pd
import gravity as grav



class TestUnconstrained(unittest.TestCase):
    """Unconstrained class for unit tests"""
    def setUp(self):
        self.f = np.array([56, 100.8, 173.6, 235.2, 87.36,
                           28., 100.8, 69.44, 235.2, 145.6,
                           22., 26.4, 136.4, 123.2, 343.2,
                           14., 75.6, 130.2, 70.56, 163.8,
                           22, 59.4,  204.6,  110.88,  171.6])
        self.V = np.repeat(np.array([56, 56, 44, 42, 66]), 5)
        self.o = np.repeat(np.array(range(1, 6)), 5)
        self.W = np.tile(np.array([10, 18, 62, 84, 78]), 5)
        self.d = np.tile(np.array(range(1, 6)), 5)
        self.dij = np.array([10, 10, 20, 20, 50,
                             20, 10, 50, 20, 30,
                             20, 30, 20, 30, 10,
                             30, 10, 20, 50, 20,
                             30, 20, 20, 50, 30])
        self.dt = pd.DataFrame({'origins': self.o,
                                'destinations': self.d,
                                'V': self.V,
                                'W': self.W,
                                'Dij': self.dij,
                                'flows': self.f})

    def test_Uconstrained(self):
        model = grav.Unconstrained(self.dt, 'origins', 'destinations', 'flows',
            ['V'], ['W'], 'Dij', 'pow')
        V = 1.0
        W = 1.0
        beta = -1.0
        self.assertAlmostEqual(model.p['V'], V, delta=.0001)
        self.assertAlmostEqual(model.p['W'], W, delta=.0001)
        self.assertAlmostEqual(model.p['beta'], beta, delta=.0001)


class TestProductionConstrained(unittest.TestCase):
    """Production constrained class for unit tests"""
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

    def test_Production_Constrained(self):
        model = grav.ProductionConstrained(self.dt, 'origins', 'destinations', 'flows',
            ['pop'], 'Dij', 'pow')
        pop = 0.7818262
        beta = -0.7365098
        self.assertAlmostEqual(model.p['pop'], pop, delta=.0001)
        self.assertAlmostEqual(model.p['beta'], beta, delta=.0001)


class TestAttractionConstrained(unittest.TestCase):
    """Attraction constrained class for unit tests"""
    def setUp(self):
        self.f = np.array([56, 100.8, 173.6, 235.2, 87.36,
                           28., 100.8, 69.44, 235.2, 145.6,
                           22., 26.4, 136.4, 123.2, 343.2,
                           14., 75.6, 130.2, 70.56, 163.8,
                           22, 59.4,  204.6,  110.88,  171.6])
        self.V = np.repeat(np.array([56, 56, 44, 42, 66]), 5)
        self.o = np.repeat(np.array(range(1, 6)), 5)
        self.W = np.tile(np.array([10, 18, 62, 84, 78]), 5)
        self.d = np.tile(np.array(range(1, 6)), 5)
        self.dij = np.array([10, 10, 20, 20, 50,
                             20, 10, 50, 20, 30,
                             20, 30, 20, 30, 10,
                             30, 10, 20, 50, 20,
                             30, 20, 20, 50, 30])
        self.dt = pd.DataFrame({'origins': self.o,
                                'destinations': self.d,
                                'V': self.V,
                                'Dij': self.dij,
                                'flows': self.f})

    def test_Attraction_Constrained(self):
        model = grav.AttractionConstrained(self.dt, 'origins', 'destinations', 'flows',
            ['V'], 'Dij', 'pow')
        V = 1.0
        beta = -1.0
        self.assertAlmostEqual(model.p['V'], V, delta=.0001)
        self.assertAlmostEqual(model.p['beta'], beta, delta=.0001)


class TestDoublyConstrained(unittest.TestCase):
    """Doubly constrained class for unit tests"""
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

    def test_Doubly_Constrained(self):
        model = grav.DoublyConstrained(self.dt, 'Origin', 'Destination', 'flows', 'Dij', 'exp')
        beta = -0.0007369
        self.assertAlmostEqual(model.p['beta'], beta, delta=.0000001)



if __name__ == '__main__':
    unittest.main()