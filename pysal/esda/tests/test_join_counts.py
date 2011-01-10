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
        jc = Join_Counts(self.y, self.w)
        self.assertAlmostEquals(jc.bb, 10.0)
        self.assertAlmostEquals(jc.zbb, 1.2060453783110545)
        self.assertAlmostEquals(jc.bw, 4.0)
        self.assertAlmostEquals(jc.zbw, -3.2659863237109046)
        self.assertAlmostEquals(jc.Ebw, 12.0 ) 
        self.assertAlmostEquals(jc.bw, 4.0)
        self.assertAlmostEquals(jc.Vbw, 6.0)
        self.assertAlmostEquals(np.sqrt(jc.Vbw), 2.4494897427831779 )

