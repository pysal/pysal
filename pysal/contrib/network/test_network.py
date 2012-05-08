"""network unittest"""
import unittest
import network as pynet
import random
random.seed(10)
import pysal

class Network_Tester(unittest.TestCase):

    def setUp(self):
        self.net = 'streets.shp'
        self.G = pynet.read_network(self.net)
        self.G2 = {(1,1): {(2,2): 0.125, (3,3): 0.75}, (2,2): {(1,1): 0.125, (4,4): 1.2},
                   (3,3): {(1,1): 0.75, (4,4): 0.375},
                   (4,4): {(2,2): 1.2, (3,3): 0.375, (5,5): 0.5},
                   (5,5): {(4,4): 0.5}, (6,6):{(7,7):1.0}, (7,7):{(6,6):1.0}}
        self.GDirected = {(1,1): {(2,2): 0.125, (3,3): 0.75}, (2,2): {(1,1): 0.125},
                   (3,3): {(1,1): 0.75, (4,4): 0.375},
                   (4,4): {(2,2): 1.2, (3,3): 0.375, (5,5): 0.5},
                   (5,5): {(4,4): 0.5}, (6,6):{(7,7):1.0}, (7,7):{(6,6):1.0}}
        self.points = [((4,4), (2,2), 0.51466686561013752, 0.68533313438986243), 
                       ((4,4), (3,3), 0.077286837052313151, 0.29771316294768685), 
                       ((6,6), (7,7), 0.82358887253344548, 0.17641112746655452)]

    def test_no_nodes(self):
        self.assertEqual(pynet.no_nodes(self.G), 230)

    def test_no_edges(self):
        self.assertEqual(pynet.no_edges(self.G), 303)

    def test_tot_net_length(self):
        self.assertAlmostEqual(pynet.tot_net_length(self.G), 52207.04600797734, places=1)

    def test_walk(self):
        correct_path = {(5, 5): {(4, 4): 0.5}, (2, 2): {(4, 4): 1.2, (1, 1): 0.125}, 
                        (1, 1): {(3, 3): 0.75, (2, 2): 0.125}, 
                        (4, 4): {(5, 5): 0.5, (3, 3): 0.375, (2, 2): 1.2}, 
                        (3, 3): {(4, 4): 0.375, (1, 1): 0.75}}
        traversal_path = pynet.walk(self.G2, (1,1))
        self.assertEqual(traversal_path, correct_path)        

    def test_components(self):
        components = pynet.components(self.G)
        self.assertEqual(pynet.no_nodes(components[0]), 230)        
        components = pynet.components(self.G2)
        self.assertEqual(len(components), 2)        

    def test_no_components(self):
        no_components = pynet.no_components(self.G2)
        self.assertEqual(no_components, 2)        

    def test_net_global_stats(self):
        stats = ['v', 'e', 'L', 'p', 'u', 'alpha', 'beta', 'emax', 'gamma', 'eta', 
                 'net_den', 'detour']
        values = pynet.net_global_stats(self.G, detour=True)
        values = dict(zip(stats, values))
        self.assertEqual(values['v'], 230)
        self.assertEqual(values['e'], 303)
        self.assertAlmostEqual(values['L'], 52207.04600797734, places=1)
        self.assertAlmostEqual(values['eta'], 172.30048187451268)
        self.assertAlmostEqual(values['beta'], 1.317391304347826, places=2)
        self.assertAlmostEqual(values['emax'], 684)
        self.assertAlmostEqual(values['gamma'], 0.44298245614035087, places=2)
        self.assertAlmostEqual(values['detour'], 0.78002937059822186, places=4)
        for k in values:
            print k, values[k]

    def test_random_projs(self):
        random.seed(10)
        random_points = pynet.random_projs(self.G2, 3)
        points = [((7, 7), (6, 6), 0.42888905467511462, 0.57111094532488538), 
                  ((7, 7), (6, 6), 0.20609823213950174, 0.79390176786049826), 
                  ((1, 1), (3, 3), 0.61769165440008411, 0.13230834559991589)] 
        self.assertEqual(random_points, points)
        
    def test_proj_distances_undirected(self):
        source = self.points[0]
        destinations = self.points[1:]
        distances = pynet.proj_distances_undirected(self.G2, source, destinations, r=1.0)
        self.assertAlmostEqual(distances.values()[0], 0.59195370266245062)

    def test_proj_distances_directed(self):
        source = self.points[0]
        destinations = self.points
        distances = pynet.proj_distances_directed(self.G2, source, destinations)
        distance_values = [0.0, 1.9626199714421757]
        self.assertEqual(distances.values(), distance_values)

    def test_dijkstras(self):
        distances = pynet.dijkstras(self.G2, (1,1))
        distance_values = {(5, 5): 1.625, (2, 2): 0.125, (1, 1): 0, (4, 4): 1.125, (3, 3): 0.75}
        self.assertEqual(distances, distance_values)

    def test_snap(self):
        snapper = pynet.Snapper(self.G2)
        projected_point = snapper.snap((2.5,2.5))
        self.assertEqual(projected_point, ((2, 2),(4, 4),3.9252311467094367e-17,1.1775693440128314e-16))

    def test_network_from_endnodes(self):
        shape = pysal.open(self.net)
        dbf = pysal.open(self.net[:-3] + 'dbf')
        def weight(geo_object, record):
            return 1
        graph = pynet.network_from_endnodes(shape, dbf, weight)
        neighbors = {(724432.38723173144, 877800.08747069736): 1, 
                     (725247.70571468933, 877812.36851842562): 1}
        start_point = (724587.78057580709, 877802.4281426128)
        self.assertEqual(graph[start_point], neighbors)

    def test_network_from_allvertices(self):
        shape = pysal.open(self.net)
        dbf = pysal.open(self.net[:-3] + 'dbf')
        graph = pynet.network_from_allvertices(shape, dbf)
        neighbors = {(724432.38723173144, 877800.08747069736): 155.41097171058956, 
                     (725247.70571468933, 877812.36851842562): 660.00000000003809}
        start_point = (724587.78057580709, 877802.4281426128)
        self.assertEqual(graph[start_point], neighbors)

    def test_read_hierarchical_network(self):
        shape = pysal.open(self.net)
        dbf = pysal.open(self.net[:-3] + 'dbf')
        graph_detail, graph_endnode, link = pynet.read_hierarchical_network(shape, dbf)
        neighbors = {(725220.77363919443, 880985.09708712087): 659.99999999994725, 
                     (724400.64597190416, 880984.45593258401): 160.12791790928244}
        vertex1 = (724560.77384088072, 880984.58111639845)
        vertex2 = (724400.64597190416, 880984.45593258401)
        self.assertEqual(graph_detail[vertex1], neighbors)
        self.assertEqual(graph_endnode[vertex1], neighbors)
        self.assertEqual(link[(vertex1, vertex2)][0], (vertex1, vertex2))

    def test_read_network(self):
        graph = pynet.read_network(self.net)
        self.assertEqual(graph, self.G) 

    def test_proj_pnt_coor(self):
        point = pynet.proj_pnt_coor(self.points[0])
        point_value = (3.6360755692750462, 3.6360755692750462)
        self.assertEqual(point, point_value)

    def test_inject_points(self):
        graph, point_coors = pynet.inject_points(self.G2, self.points)
        self.assertEqual(len(graph), len(self.G2) + 3)

    def test_mesh_network(self):
        graph = pynet.mesh_network(self.G2, 0.1)
        self.assertEqual(len(graph), 41)

    def test_write_network_to_pysalshp(self):
        out = 'output_network.shp'
        pynet.write_network_to_pysalshp(self.G, out)
        graph = pynet.read_network(out)
        self.assertEqual(graph, self.G)    

    def test_write_valued_network_to_shp(self):
        out = 'output_network.shp'
        fields = ['WEIGHT1','WEIGHT2']
        types = [('N',7,3),('N',7,3)]
        values = {(1,1):1,(2,2):2,(3,3):3,(4,4):4,(5,5):5,(6,6):6,(7,7):7}
        def doubleX(values, node):
            return values[node]*2.0
        pynet.write_valued_network_to_shp(out,fields,types,self.G2,values,doubleX)
        graph = pynet.read_network(out)
        self.assertEqual(len(graph), len(self.G2))    

    def test_list_network_to_shp(self):
        out = 'output_network.shp'
        fields = ['WEIGHT']
        types = [('N',7,3)]
        list_network = []
        for node1 in self.G2:
            for node2 in self.G2[node1]:
                list_network.append(((node1, node2),random.random()))
        pynet.write_list_network_to_shp(out,fields,types,list_network)
        graph = pynet.read_network(out)
        self.assertEqual(len(graph), len(self.G2))    


suite = unittest.TestSuite()
test_classes = [Network_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
