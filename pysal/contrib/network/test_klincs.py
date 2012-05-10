"""network unittest"""
import unittest
import network as pynet
import klincs 
import random
random.seed(10)
import pysal
import numpy as np

class KLINCS_Tester(unittest.TestCase):

    def setUp(self):
        self.population = range(5)
        self.weights = [0.1, 0.25, 0.1, 0.2, 0.35]
        np.random.seed(10)
        self.network_file = 'streets.shp'
        self.G = pynet.read_network(self.network_file)
        self.references = [[i, [n]] for i, n in enumerate(self.G.keys())]
        self.scale_set = (0, 1500, 500) 
        self.network_distances_cache = {}
        search_radius = self.scale_set[1]
        for i, node in self.references:
            n = node[0]
            self.network_distances_cache[n] = pynet.dijkstras(self.G, n, search_radius) 
        self.snapper = pynet.Snapper(self.G)
        self.events_file = 'crimes.shp' 
        points = self.get_points_from_shapefile(self.events_file)
        self.events = []
        for p in points:
            self.events.append(self.snapper.snap(p[0]))
        self.test_node = (724587.78057580709, 877802.4281426128)

    def get_points_from_shapefile(self, src_filename, src_uid=None):
        src_file = pysal.open(src_filename)
        dbf = pysal.open(src_filename[:-3] + 'dbf')
        src_uid_index = dbf.header.index(src_uid) if src_uid else None 
        if src_uid_index != None:
            def get_index(index, record):
                return record[src_uid_index]
        else:
            def get_index(index, record):
                return index
        if src_file.type == pysal.cg.shapes.Polygon:
            def get_geom(g):
                return g.centroid
        elif src_file.type == pysal.cg.shapes.Point:
            def get_geom(g):
                return (g[0], g[1])
        srcs = [] 
        for i, rec in enumerate(src_file):
            srcs.append([get_geom(rec), get_index(i, dbf.next())]) 
        src_file.close()
        return srcs 

    def test_WeightedRandomSampleGenerator(self):
        generator = klincs.WeightedRandomSampleGenerator(self.weights, self.population, 3)
        sample = generator.next()
        self.assertEqual(sample,[4, 0, 3])

    def test_RandomSampleGenerator(self):
        generator = klincs.RandomSampleGenerator(self.population, 3)
        sample = generator.next()
        self.assertEqual(sample,[2, 1, 3])

    def test_local_k(self):
        network = self.G
        references = self.references
        events = self.events
        scale_set = self.scale_set 
        cache = self.network_distances_cache
        node2localK, net_distances = klincs.local_k(network, events, references, scale_set, cache)
        # node2localK
        # for each reference node,
        # local_k returns the number of events within a distance 
        # the distance is determined by scale_set
        # example: (724587.78057580709, 877802.4281426128): {0: 0, 1000: 22, 500: 9}
        # 
        # net_distances - a dictionary containing network distances 
        # between nodes in the input network
        test_node = self.test_node
        self.assertEqual(node2localK[test_node][500], 9)

    def test_cluster_type(self):
        cluster1 = klincs.cluster_type(0.25, 0.01, 0.33)
        cluster2 = klincs.cluster_type(0.45, 0.01, 0.33)
        self.assertEqual(cluster1, 0)
        self.assertEqual(cluster2, 1)

    def test_simulate_local_k_01(self):
        sims = 1
        n = len(self.events)
        net_file = self.network_file
        network = self.G
        events = self.events
        refs = self.references
        scale_set = self.scale_set
        cache = self.network_distances_cache
        args = (sims, n, net_file, network, events, refs, scale_set, cache)
        sim_outcomes = klincs.simulate_local_k_01(args)
        self.assertEqual(sim_outcomes[0][self.test_node], {0: 0, 1000: 9, 500: 4})

    def test_simulate_local_k_02(self):
        sims = 1
        n = 50
        refs = self.references
        scale_set = self.scale_set
        cache = self.network_distances_cache
        args = (sims, n, refs, scale_set, cache)
        sim_outcomes = klincs.simulate_local_k_02(args)
        self.assertEqual(sim_outcomes[0][self.test_node], {0: 1, 1000: 2, 500: 2})
          
    def test_k_cluster(self):
        network = self.G
        events = self.events
        refs = self.references
        scale_set = self.scale_set
        sims = 1
        sim_network = self.network_file
        localKs = klincs.k_cluster(network, events, refs, scale_set, sims, sim_network=sim_network)
        test_node = self.test_node
        # in [9,4,4,1]
        # 9 - the number of events observed at the search radius of 500
        # 4 - the lowest number of simulated points observed at the search radius of 500
        # 4 - the highest number of simulated points observed at the search radius of 500
        # 1 - the type of cluster
        self.assertEqual(localKs[test_node][500],[9,4,4,1])

suite = unittest.TestSuite()
test_classes = [KLINCS_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
