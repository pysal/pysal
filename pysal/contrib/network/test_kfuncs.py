"""network unittest"""
import unittest
import network as pynet
import kfuncs

class Kfuncs_Tester(unittest.TestCase):

    def setUp(self):
        self.distances = {1:[1,2,3,4],2:[1,1,2,3],3:[2,1,1,2],
                          4:[3,2,1,1],5:[4,3,2,1]}

    def test__fxrange(self):
        values = kfuncs._fxrange(0.0,1.0,0.2)
        for v1, v2 in zip(values, [0.0,0.2,0.4,0.6,0.8,1.0]):
            self.assertAlmostEqual(v1, v2)

    def test__binary_search(self):
        v = kfuncs._binary_search([0.0,0.2,0.4,0.6,0.8,1.0],0.9)
        self.assertEqual(v, 5)

    def test_kt_values(self):
        expected_values = {1: {0.5: 0, 1.5: 10, 2.5: 20}, 
                           2: {0.5: 0, 1.5: 20, 2.5: 30}, 
                           3: {0.5: 0, 1.5: 20, 2.5: 40}, 
                           4: {0.5: 0, 1.5: 20, 2.5: 30}, 
                           5: {0.5: 0, 1.5: 10, 2.5: 20}}
        kfunc_values = {}
        for k, v in self.distances.items():
            kfunc_values[k] = kfuncs.kt_values((0.5,3.5,1.0),v,10)
        self.assertEqual(kfunc_values, expected_values)

suite = unittest.TestSuite()
test_classes = [Kfuncs_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
