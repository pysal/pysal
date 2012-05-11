"""network unittest"""
import unittest
import network as pynet
import numpy as np
import weights 

class Weights_Tester(unittest.TestCase):

    def test_dist_weights(self):
        ids = np.array(map(str,range(1,9)))
        w = weights.dist_weights('distances.csv','knn',ids,3)
        self.assertEqual(w.neighbors['1'],['6','8','7'])

suite = unittest.TestSuite()
test_classes = [Weights_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
