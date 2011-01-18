
from pysal.esda import mapclassify
import doctest
import unittest
import numpy

class _TestMapclassify(unittest.TestCase):
    def setUp(self):
        vals = [23.0, 3.0, 29.0, 2.0, 6.0, 1.0, 2.0, 381.0, 515.0, 333.0, 3.0, 3.0, 8.0, 120.0, 3.0, 26.0, 1.0, 9.0, 42.0, 5.0, 1.0, 1.0, 2258.0, 1.0, 18.0, 1.0, 1501.0, 4.0, 9.0, 3.0, 3332.0, 297.0, 1.0, 499.0, 61.0, 1.0, 16.0, 67.0, 2.0, 8.0, 61.0, 34.0, 56.0, 15.0, 2.0, 14.0, 13.0, 110.0, 67.0, 1.0, 895.0, 2.0, 9.0, 646.0, 4.0, 1.0, 176.0, 17.0, 1.0, 1.0, 2.0, 14.0, 20.0, 121.0, 8.0, 355.0, 326.0, 125.0, 901.0, 5.0, 2.0, 279.0, 16.0, 5.0, 47.0, 424.0, 6.0, 6.0, 2.0, 4.0, 7.0, 23.0, 1.0, 56.0, 2.0, 11.0, 2.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 7.0, 3.0, 30.0, 2.0, 4.0, 3.0, 1.0, 588.0, 75.0, 7.0, 4.0, 1.0, 13.0, 360.0, 71.0, 7.0, 4.0, 1.0, 4.0, 163.0, 8.0, 94.0, 33.0, 410.0, 14.0, 293.0, 1.0, 8.0, 15.0, 3.0, 89.0, 97.0, 25.0, 78.0, 13.0, 1.0, 153.0, 14.0, 33.0, 39.0, 932.0, 1.0, 203.0, 300.0, 4.0, 124.0, 31.0, 235.0, 6.0, 9.0, 1647.0, 19.0, 14752.0, 3.0, 35.0, 1.0, 77.0, 38.0, 1.0, 4.0, 1.0, 2.0, 5.0]
        self.V = numpy.array(vals)

    def test_Natural_Breaks(self):
        for i in range(10):
            nb = mapclassify.Natural_Breaks(self.V,5)
            self.assertEquals(nb.k,5)
            self.assertEquals(len(nb.counts),5)
        

suite = unittest.TestLoader().loadTestsFromTestCase(_TestMapclassify)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
