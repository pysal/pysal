from __future__ import division
import unittest

import numpy as np
import pysal as ps

from pysal.network import util

class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.ntw = ps.Network(ps.examples.get_path('streets.shp'))

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

    def test_distance_band_weights(self):
        #I do not trust this result, test should be manually checked.
        w = self.ntw.distancebandweights(threshold=500)
        self.assertEqual(w.n, 230)
        self.assertEqual(w.histogram,
                         [(1, 22), (2, 58), (3, 63), (4, 40),
                          (5, 36), (6, 3), (7, 5), (8, 3)])

    def test_edge_segmentation(self):
        n200 = self.ntw.segment_edges(200.0)
        self.assertEqual(len(n200.edges), 688)
        n200 = None

    def test_enum_links_node(self):
        coincident = self.ntw.enum_links_node(24)
        self.assertIn((24,48), coincident)

class TestNetworkPointPattern(unittest.TestCase):

    def setUp(self):
        self.ntw = ps.Network(ps.examples.get_path('streets.shp'))
        for obs in ['schools', 'crimes']:
            self.ntw.snapobservations(ps.examples.get_path('{}.shp'.format(obs)), obs, attribute=True)
            setattr(self, obs, self.ntw.pointpatterns[obs])

    def tearDown(self):
        pass

    def test_add_point_pattern(self):
        self.assertEqual(self.crimes.npoints, 287)
        self.assertIn('properties', self.crimes.points[0])
        self.assertIn([1,1],self.crimes.points[0]['properties'])

    def test_count_per_edge(self):
        counts = self.ntw.count_per_edge(self.ntw.pointpatterns['crimes'].obs_to_edge,
                                         graph=False)
        meancounts = sum(counts.values()) / float(len(counts.keys()))
        self.assertAlmostEqual(meancounts, 2.682242, places=5)

    def test_count_per_graph_edge(self):
        counts = self.ntw.count_per_edge(self.ntw.pointpatterns['crimes'].obs_to_edge,
                                         graph=True)
        meancounts = sum(counts.values()) / float(len(counts.keys()))
        self.assertAlmostEqual(meancounts, 3.29885, places=5)

    def test_simulate_normal_observations(self):
        npoints = self.ntw.pointpatterns['crimes'].npoints
        sim = self.ntw.simulate_observations(npoints)
        self.assertEqual(npoints, sim.npoints)

    def test_simulate_poisson_observations(self):
        pass

    def test_all_neighbor_distances(self):
        distancematrix = self.ntw.allneighbordistances(self.schools)
        self.assertAlmostEqual(np.nansum(distancematrix[0]), 17682.436988, places=4)

        for k, (distances, predlist) in self.ntw.alldistances.iteritems():
            self.assertEqual(distances[k], 0)

            # turning off the tests associated with util.generatetree() for now,
            # these can be restarted if that functionality is used in the future 
            #for p, plists in predlist.iteritems():
            #    self.assertEqual(plists[-1], k)

            #self.assertEqual(self.ntw.node_list, predlist.keys())

    def test_nearest_neighbor_distances(self):

        with self.assertRaises(KeyError):
            self.ntw.nearestneighbordistances('i_should_not_exist')

        nnd = self.ntw.nearestneighbordistances('schools')
        nnd2 = self.ntw.nearestneighbordistances('schools',
                                                 'schools')
        np.testing.assert_array_equal(nnd, nnd2)

    def test_nearest_neighbor_search(self):
        pass


class TestNetworkUtils(unittest.TestCase):

    def setUp(self):
        self.ntw = ps.Network(ps.examples.get_path('streets.shp'))

    def test_dijkstra(self):
        self.distance, self.pred = util.dijkstra(self.ntw, self.ntw.edge_lengths, 0)
        self.assertAlmostEqual(self.distance[196], 5505.668247, places=4)
        self.assertEqual(self.pred[196], 133)

if __name__ == 'main':
    unittest.main()
