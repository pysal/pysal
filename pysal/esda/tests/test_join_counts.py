import unittest
import numpy as np

from ..join_counts import Join_Counts
from ...weights import lat2W
from ...common import pandas

PANDAS_EXTINCT = pandas is None

class Join_Counts_Tester(unittest.TestCase):
    """Unit test for Join Counts"""
    def setUp(self):
        self.w = lat2W(4, 4)
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
    
    @unittest.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_by_col(self):
        import pandas as pd
        df = pd.DataFrame(self.y, columns=['y'])
        np.random.seed(12345)
        r1 = Join_Counts.by_col(df, ['y'], w=self.w, permutations=999)
        
        bb = np.unique(r1.y_bb.values)
        bw = np.unique(r1.y_bw.values)
        bb_p = np.unique(r1.y_p_sim_bb.values)
        bw_p = np.unique(r1.y_p_sim_bw.values)
        np.random.seed(12345)
        c = Join_Counts(self.y, self.w, permutations=999)
        self.assertAlmostEquals(bb, c.bb)
        self.assertAlmostEquals(bw, c.bw)
        self.assertAlmostEquals(bb_p, c.p_sim_bb)
        self.assertAlmostEquals(bw_p, c.p_sim_bw)

suite = unittest.TestSuite()
test_classes = [Join_Counts_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
