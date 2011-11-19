"""Geary Unittest."""
import unittest
import pysal
from pysal.esda import geary 
import numpy as np

class Geary_Tester(unittest.TestCase):
    """Geary class for unit tests."""
    def setUp(self):
        self.w = pysal.open("pysal/examples/book.gal").read()
        f = pysal.open("pysal/examples/book.txt")
        self.y = np.array(f.by_col['y'])

    def test_Geary(self):
        c = geary.Geary(self.y, self.w, permutations=0)
        self.assertAlmostEquals(c.C, 0.33281733746130032 )
        self.assertAlmostEquals(c.EC, 1.0 )
        
        self.assertAlmostEquals(c.VC_norm, 0.035539215686274508 )
        self.assertAlmostEquals(c.p_norm, 0.00020075926879692396 )
        self.assertAlmostEquals(c.z_norm, -3.5390836893031601 )
        self.assertAlmostEquals(c.seC_norm, 0.1885184757159746 )

        self.assertAlmostEquals(c.VC_rand, 0.042979552532918339 )
        self.assertAlmostEquals(c.p_rand, 0.0006449762328940567 )
        self.assertAlmostEquals(c.z_rand, -3.2182057564509572 )
        self.assertAlmostEquals(c.seC_rand, 0.20731510444952711 )

        np.random.seed(100)
        c = geary.Geary(self.y, self.w, permutations=99)
        self.assertAlmostEquals(c.C, 0.33281733746130032 )
        self.assertAlmostEquals(c.EC, 1.0 )
        
        self.assertAlmostEquals(c.VC_norm, 0.035539215686274508 )
        self.assertAlmostEquals(c.p_norm, 0.00020075926879692396 )
        self.assertAlmostEquals(c.z_norm, -3.5390836893031601 )
        self.assertAlmostEquals(c.seC_norm, 0.1885184757159746 )

        self.assertAlmostEquals(c.VC_rand, 0.042979552532918339 )
        self.assertAlmostEquals(c.p_rand, 0.0006449762328940567 )
        self.assertAlmostEquals(c.z_rand, -3.2182057564509572 )
        self.assertAlmostEquals(c.seC_rand, 0.20731510444952711 )
    
        self.assertAlmostEquals(c.EC_sim, 0.98686555962097777 )
        self.assertAlmostEquals(c.VC_sim, 0.024409129957879146 )
        self.assertAlmostEquals(c.p_sim, 0.01 )
        self.assertAlmostEquals(c.p_z_sim, 1.4174958245738445e-05 )
        self.assertAlmostEquals(c.z_sim, -4.1863315399605332 )
        self.assertAlmostEquals(c.seC_sim, 0.1562342150678882 )


suite = unittest.TestSuite()
test_classes = [ Geary_Tester ]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
