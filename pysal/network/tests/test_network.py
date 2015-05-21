import pysal as ps
import unittest

class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.ntw = ps.Network(ps.examples.get_path('geodanet/streets.shp'))

    def tearDown(self):
        pass

    def test_extract_network(self):
        self.assertEqual(len(self.ntw.edges), 303)
        self.assertEqual(len(self.ntw.nodes), 230)

        edgelengths = self.ntw.edge_lengths.values()
        self.assertAlmostEqual(sum(edgelengths), 104414.0920159, places=5)


        self.assertIn(0,self.ntw.adjacencylist[1])
        self.assertIn(0, self.ntw.adjacencylist[2])
        self.assertNotIn(0, self.ntw.adjacencylist[3])

    def test_contiguity_weights(self):
        w = self.ntw.contiguityweights(graph=False)

        self.assertEqual(w.n, 303)
        self.assertEqual(w.histogram,
                         [(2, 35), (3, 89), (4, 105), (5, 61), (6, 13)])

    def test_contiguity_weights_graph(self):
        w = self.ntw.contiguityweights(graph=True)

        self.assertEqual(w.n, 179)
        self.assertEqual(w.histogram,
                         [(2, 2), (3, 2), (4, 45), (5, 82), (6, 48)])
