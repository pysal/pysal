import unittest
import numpy as np
import pysal
from pysal.esda.join_counts import Join_Counts


class Join_Counts_Tester(unittest.TestCase):
    """Unit test for Join Counts"""
    def setUp(self):
        self.w = pysal.lat2W(4, 4)
        self.y = np.ones(16)
        self.y[0:8] = 0

    def test_Join_Counts(self):
        """Test method"""
        np.random.seed(12345)
        jc = Join_Counts(self.y, self.w)
        self.assertAlmostEquals(jc.bb, 10.0)
        self.assertAlmostEquals(jc.bw, 4.0)
        self.assertAlmostEquals(jc.ww, 10.0)
        self.assertAlmostEquals(jc.J, 24.0)
        self.assertAlmostEquals(len(jc.sim_bb), 999)
        self.assertAlmostEquals(jc.p_sim_bb, 0.0030000000000000001)
        self.assertAlmostEquals(np.mean(jc.sim_bb), 5.5465465465465469)
        self.assertAlmostEquals(np.max(jc.sim_bb), 10.0)
        self.assertAlmostEquals(np.min(jc.sim_bb), 0.0)
        self.assertAlmostEquals(len(jc.sim_bw), 999)
        self.assertAlmostEquals(jc.p_sim_bw, 1.0)
        self.assertAlmostEquals(np.mean(jc.sim_bw), 12.811811811811811)
        self.assertAlmostEquals(np.max(jc.sim_bw), 24.0)
        self.assertAlmostEquals(np.min(jc.sim_bw), 7.0)

suite = unittest.TestSuite()
test_classes = [Join_Counts_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
