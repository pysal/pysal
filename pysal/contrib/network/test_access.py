"""access unittest"""
import unittest
import access as pyacc

class Access_Tester(unittest.TestCase):

    def setUp(self):
        self.distances = {1:[1,2,3,4],2:[1,1,2,3],3:[2,1,1,2],
                          4:[3,2,1,1],5:[4,3,2,1]}

    def test_coverage(self):
        coverage = []
        for d in self.distances.values():
            coverage.append(pyacc.coverage(d, 2.5))
        self.assertEqual(coverage, [2,3,4,3,2])

    def test_equity(self):
        equity = []
        for d in self.distances.values():
            equity.append(pyacc.equity(d))
        self.assertEqual(equity, [1,1,1,1,1])

    def test_potential_entropy(self):
        entropy = []
        for d in self.distances.values():
            entropy.append(pyacc.potential_entropy(d))
        entropy_values = [0.57131743166465321, 0.92088123394736132, 
                1.0064294488161101, 0.92088123394736132, 0.57131743166465321]
        self.assertEqual(entropy, entropy_values)

    def test_potential_gravity(self):
        gravity = []
        for d in self.distances.values():
            gravity.append(pyacc.potential_gravity(d))
        gravity_values = [1.4236111111111112, 2.3611111111111112, 2.5, 
                          2.3611111111111112, 1.4236111111111112]
        self.assertEqual(gravity, gravity_values)

    def test_travel_cost(self):
        cost = []
        for d in self.distances.values():
            cost.append(pyacc.travel_cost(d))
        self.assertEqual(cost, [10, 7, 6, 7, 10])

suite = unittest.TestSuite()
test_classes = [Access_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
