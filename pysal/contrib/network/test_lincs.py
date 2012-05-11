"""network unittest"""
import unittest
import network as pynet
import lincs 
import random
random.seed(10)
import pysal
import numpy as np
import copy

class LINCS_Tester(unittest.TestCase):

    def setUp(self):
        self.base = np.array([100, 80, 50, 120, 90])
        self.events = np.array([20, 20, 5, 10, 25])
        np.random.seed(10)
        self.observed = np.array([0.1,0.15,0.2])
        self.simulated = np.array([[0.05,0.10,0.25],[0.12,0.11,0.3],[0.11,0.09,0.27]])
        self.network_file = 'streets.shp'
        self.G = pynet.read_network(self.network_file)
        self.test_link = ((724432.38723173144, 877800.08747069736), 
                          (724587.78057580709, 877802.4281426128))
        self.G2 = copy.deepcopy(self.G)
        done = set()
        for n1 in self.G2:
            for n2 in self.G2[n1]:
                if (n1, n2) in done:
                    continue
                dist = self.G2[n1][n2]
                base = int(random.random()*1000)
                event = int(random.random()*base)
                self.G2[n1][n2] = [dist, base, event]
                self.G2[n2][n1] = [dist, base, event]
                done.add((n1,n2))
                done.add((n2,n1))

    def test_unconditional_sim(self):
        simulated_events = lincs.unconditional_sim(self.events, self.base, 2)
        self.assertEqual(list(simulated_events[0]),[21,15])

    def test_unconditional_sim_poisson(self):
        simulated_events = lincs.unconditional_sim_poisson(self.events, self.base, 2)
        self.assertEqual(list(simulated_events[0]),[22,21])

    def test_conditional_multinomial(self):
        simulated_events = lincs.conditional_multinomial(self.events, self.base, 2)
        self.assertEqual(list(simulated_events[0]),[21,18])

    def test_pseudo_pvalues(self):
        pseudo_pvalues = lincs.pseudo_pvalues(self.observed, self.simulated)
        self.assertEqual(list(pseudo_pvalues[0]),[ 0.5,  0.5,  0.5])

    def test_node_weights(self):
        w, id2link = lincs.node_weights(self.G)
        self.assertEqual(w.neighbors[0],[1, 2, 3, 4])

    def test_edgepoints_from_network(self):
        id2linkpoints, id2attr, link2id = lincs.edgepoints_from_network(self.G)
        link = id2linkpoints[0]
        self.assertEqual(link[:2], self.test_link)
        self.assertEqual(link2id[link[:2]], 0)

    def test_dist_weights(self):
        id2linkpoints, id2attr, link2id = lincs.edgepoints_from_network(self.G)
        w, id2link = lincs.dist_weights(self.G, id2linkpoints, link2id, 500)
        self.assertEqual(w.neighbors[0],[1,154,153,155])
        self.assertEqual(id2link[0], self.test_link)

    def test_lincs(self):
        network = self.G2
        event_index = 2
        base_index = 1
        weight = 'Distance-based'
        dist = 500
        lisa_func = 'moran'
        sim_method = 'permutations'
        sim_num = 2
        lisa, w = lincs.lincs(network, event_index, base_index, weight, dist, lisa_func, sim_method, sim_num)
        self.assertEqual(lisa[0][3], -0.64342055427251854)

suite = unittest.TestSuite()
test_classes = [LINCS_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
