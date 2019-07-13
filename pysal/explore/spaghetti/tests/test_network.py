import unittest
import numpy as np
from pysal.lib import cg, examples

# dev import structure
from .. import network as spgh
from .. import util

try:
    import geopandas
    GEOPANDAS_EXTINCT = False
except ImportError:
    GEOPANDAS_EXTINCT = True


class TestNetwork(unittest.TestCase):
    
    def setUp(self):
        self.path_to_shp = examples.get_path('streets.shp')
        
        # network instantiated from shapefile
        self.ntw_from_shp = spgh.Network(in_data=self.path_to_shp,
                                         weightings=True,
                                         w_components=True)
        self.n_known_arcs, self.n_known_vertices = 303, 230
    
    def tearDown(self):
        pass
    
    def test_network_data_read(self):
        # shp test against known
        self.assertEqual(len(self.ntw_from_shp.arcs),
                             self.n_known_arcs)
        self.assertEqual(len(self.ntw_from_shp.vertices),
                             self.n_known_vertices)
        
        arc_lengths = self.ntw_from_shp.arc_lengths.values()
        self.assertAlmostEqual(sum(arc_lengths), 104414.0920159, places=5)
        
        self.assertIn(0, self.ntw_from_shp.adjacencylist[1])
        self.assertIn(0, self.ntw_from_shp.adjacencylist[2])
        self.assertNotIn(0, self.ntw_from_shp.adjacencylist[3])
    
    @unittest.skipIf(GEOPANDAS_EXTINCT, 'Missing Geopandas')
    def test_network_from_geopandas(self):
        # network instantiated from geodataframe
        gdf = geopandas.read_file(self.path_to_shp)
        self.ntw_from_gdf = spgh.Network(in_data=gdf,
                                         w_components=True)
        
        # gdf test against known
        self.assertEqual(len(self.ntw_from_gdf.arcs),
                             self.n_known_arcs)
        self.assertEqual(len(self.ntw_from_gdf.vertices),
                             self.n_known_vertices)
        
        # shp against gdf
        self.assertEqual(len(self.ntw_from_shp.arcs),
                         len(self.ntw_from_gdf.arcs))
        self.assertEqual(len(self.ntw_from_shp.vertices),
                         len(self.ntw_from_gdf.vertices))
    
    def test_contiguity_weights(self):
        known_network_histo = [(2, 35), (3, 89), (4, 105), (5, 61), (6, 13)]
        observed_network_histo = self.ntw_from_shp.w_network.histogram
        self.assertEqual(known_network_histo, observed_network_histo)
        
        known_graph_histo = [(2, 2), (3, 2), (4, 47), (5, 80), (6, 48)]
        observed_graph_histo = self.ntw_from_shp.w_graph.histogram
        self.assertEqual(observed_graph_histo, known_graph_histo)
    
    def test_components(self):
        known_network_arc = (225, 226)
        observed_network_arc = self.ntw_from_shp.network_component2arc[0][-1]
        self.assertEqual(observed_network_arc, known_network_arc)
        
        known_graph_edge = (207, 208)
        observed_graph_edge = self.ntw_from_shp.graph_component2edge[0][-1]
        self.assertEqual(observed_graph_edge, known_graph_edge)
    
    def test_distance_band_weights(self):
        w = self.ntw_from_shp.distancebandweights(threshold=500)
        self.assertEqual(w.n, 230)
        self.assertEqual(w.histogram,
                         [(1, 22), (2, 58), (3, 63), (4, 40),
                          (5, 36), (6, 3), (7, 5), (8, 3)])
    
    def test_split_arcs_200(self):
        n200 = self.ntw_from_shp.split_arcs(200.0)
        self.assertEqual(len(n200.arcs), 688)
    
    def test_enum_links_vertex(self):
        coincident = self.ntw_from_shp.enum_links_vertex(24)
        self.assertIn((24, 48), coincident)
    
    @unittest.skipIf(GEOPANDAS_EXTINCT, 'Missing Geopandas')
    def test_element_as_gdf(self):
        vertices, arcs = spgh.element_as_gdf(self.ntw_from_shp,
                                             vertices=True,
                                             arcs=True)
        
        known_vertex_wkt = 'POINT (728368.04762 877125.89535)'
        obs_vertex = vertices.loc[(vertices['id'] == 0), 'geometry'].squeeze()
        obs_vertex_wkt = obs_vertex.wkt
        self.assertEqual(obs_vertex_wkt, known_vertex_wkt)
        
        known_arc_wkt = 'LINESTRING (728368.04762 877125.89535, '\
                         + '728368.13931 877023.27186)'
        obs_arc = arcs.loc[(arcs['id'] == (0,1)), 'geometry'].squeeze()
        obs_arc_wkt = obs_arc.wkt
        self.assertEqual(obs_arc_wkt, known_arc_wkt)
        
        arcs = spgh.element_as_gdf(self.ntw_from_shp, arcs=True)
        known_arc_wkt = 'LINESTRING (728368.04762 877125.89535, '\
                         + '728368.13931 877023.27186)'
        obs_arc = arcs.loc[(arcs['id'] == (0,1)), 'geometry'].squeeze()
        obs_arc_wkt = obs_arc.wkt
        self.assertEqual(obs_arc_wkt, known_arc_wkt)
    
    def test_round_sig(self):
        # round to 2 significant digits test
        x_round2, y_round2 = 1200, 1900
        self.ntw_from_shp.vertex_sig = 2
        obs_xy_round2 = self.ntw_from_shp._round_sig((1215, 1865))
        self.assertEqual(obs_xy_round2, (x_round2, y_round2))
        
        # round to no significant digits test
        x_roundNone, y_roundNone = 1215, 1865
        self.ntw_from_shp.vertex_sig = None
        obs_xy_roundNone = self.ntw_from_shp._round_sig((1215, 1865))
        self.assertEqual(obs_xy_roundNone, (x_roundNone, y_roundNone))


class TestNetworkPointPattern(unittest.TestCase):
    
    def setUp(self):
        path_to_shp = examples.get_path('streets.shp')
        self.ntw = spgh.Network(in_data=path_to_shp)
        self.pp1_str = 'schools'
        self.pp2_str = 'crimes'
        iterator = [(self.pp1_str, 'pp1'),
                    (self.pp2_str, 'pp2')]
        for (obs, idx) in iterator:
            path_to_shp = examples.get_path('%s.shp' % obs)
            self.ntw.snapobservations(path_to_shp, obs, attribute=True)
            setattr(self, idx, self.ntw.pointpatterns[obs])
        self.known_pp1_npoints = 8
    
    def tearDown(self):
        pass
    
    @unittest.skipIf(GEOPANDAS_EXTINCT, 'Missing Geopandas')
    def test_pp_from_geopandas(self):
        self.gdf_pp1_str = 'schools'
        self.gdf_pp2_str = 'crimes'
        iterator = [(self.gdf_pp1_str, 'gdf_pp1'),
                    (self.gdf_pp2_str, 'gdf_pp2')]
        for (obs, idx) in iterator:
            path_to_shp = examples.get_path('%s.shp' % obs)
            in_data = geopandas.read_file(path_to_shp)
            self.ntw.snapobservations(in_data, obs, attribute=True)
            setattr(self, idx, self.ntw.pointpatterns[obs])
        
        self.assertEqual(self.pp1.npoints, self.gdf_pp1.npoints)
        self.assertEqual(self.pp2.npoints, self.gdf_pp2.npoints)
    
    def test_split_arcs_1000(self):
        n1000 = self.ntw.split_arcs(1000.0)
        self.assertEqual(len(n1000.arcs), 303)
    
    def test_add_point_pattern(self):
        self.assertEqual(self.pp1.npoints, self.known_pp1_npoints)
        self.assertIn('properties', self.pp1.points[0])
        self.assertIn([1], self.pp1.points[0]['properties'])
    
    def test_count_per_link_network(self):
        counts = self.ntw.count_per_link(self.pp1.obs_to_arc, graph=False)
        meancounts = sum(counts.values()) / float(len(counts.keys()))
        self.assertAlmostEqual(meancounts, 1.0, places=5)
    
    def test_count_per_edge_graph(self):
        counts = self.ntw.count_per_link(self.pp1.obs_to_arc, graph=True)
        meancounts = sum(counts.values()) / float(len(counts.keys()))
        self.assertAlmostEqual(meancounts, 1.0, places=5)
    
    def test_simulate_normal_observations(self):
        sim = self.ntw.simulate_observations(self.known_pp1_npoints)
        self.assertEqual(self.known_pp1_npoints, sim.npoints)
    
    def test_simulate_poisson_observations(self):
        sim = self.ntw.simulate_observations(self.known_pp1_npoints,
                                             distribution='poisson')
        self.assertEqual(self.known_pp1_npoints, sim.npoints)
    
    def test_all_neighbor_distances(self):
        matrix1, tree = self.ntw.allneighbordistances(self.pp1_str,
                                                      gen_tree=True)
        known_mtx_val = 17682.436988
        known_tree_val = (173, 64)
        
        self.assertAlmostEqual(np.nansum(matrix1[0]), known_mtx_val, places=4)
        self.assertEqual(tree[(6, 7)], known_tree_val)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix2 = self.ntw.allneighbordistances(self.pp1_str, fill_diagonal=0.)
        observed = matrix2.diagonal()
        known = np.zeros(matrix2.shape[0])
        self.assertEqual(observed.all(), known.all())
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix3 = self.ntw.allneighbordistances(self.pp1_str, snap_dist=True)
        known_mtx_val = 3218.2597894
        observed_mtx_val = matrix3
        self.assertAlmostEqual(observed_mtx_val[0, 1], known_mtx_val, places=4)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix4 = self.ntw.allneighbordistances(self.pp1_str,
                                                fill_diagonal=0.)
        observed = matrix4.diagonal()
        known = np.zeros(matrix4.shape[0])
        self.assertEqual(observed.all(), known.all())
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix5, tree = self.ntw.allneighbordistances(self.pp2_str,
                                                      gen_tree=True)
        known_mtx_val = 1484112.694526529
        known_tree_val = (-0.1, -0.1)
        
        self.assertAlmostEqual(np.nansum(matrix5[0]), known_mtx_val, places=4)
        self.assertEqual(tree[(18, 19)], known_tree_val)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
    def test_all_neighbor_distances_multiproccessing(self):
        matrix1, tree = self.ntw.allneighbordistances(self.pp1_str,
                                                      fill_diagonal=0.,
                                                      n_processes='all',
                                                      gen_tree=True)
        known_mtx1_val = 17682.436988
        known_tree_val = (173, 64)
        
        observed = matrix1.diagonal()
        known = np.zeros(matrix1.shape[0])
        self.assertEqual(observed.all(), known.all())
        self.assertAlmostEqual(np.nansum(matrix1[0]), known_mtx1_val, places=4)
        self.assertEqual(tree[(6, 7)], known_tree_val)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix2 = self.ntw.allneighbordistances(self.pp1_str, n_processes=2)
        known_mtx2_val = 17682.436988
        self.assertAlmostEqual(np.nansum(matrix2[0]), known_mtx2_val, places=4)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        matrix3, tree = self.ntw.allneighbordistances(self.pp1_str,
                                                      fill_diagonal=0.,
                                                      n_processes=2,
                                                      gen_tree=True)
        known_mtx3_val = 17682.436988
        known_tree_val = (173, 64)
        
        self.assertAlmostEqual(np.nansum(matrix3[0]), known_mtx3_val, places=4)
        self.assertEqual(tree[(6, 7)], known_tree_val)
        del self.ntw.alldistances
        del self.ntw.distancematrix
    
    def test_nearest_neighbor_distances(self):
        # general test
        with self.assertRaises(KeyError):
            self.ntw.nearestneighbordistances('i_should_not_exist')
        nnd1 = self.ntw.nearestneighbordistances(self.pp1_str)
        nnd2 = self.ntw.nearestneighbordistances(self.pp1_str,
                                                 destpattern=self.pp1_str)
        nndv1 = np.array(list(nnd1.values()))[:,1].astype(float)
        nndv2 = np.array(list(nnd2.values()))[:,1].astype(float)
        np.testing.assert_array_almost_equal_nulp(nndv1, nndv2)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        # nearest neighbor keeping zero test
        known_zero = ([19], 0.0)[0]
        nn_c = self.ntw.nearestneighbordistances(self.pp2_str,
                                                 keep_zero_dist=True)
        self.assertEqual(nn_c[18][0], known_zero)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        # nearest neighbor omitting zero test
        known_nonzero = ([11], 165.33982412719126)[1]
        nn_c = self.ntw.nearestneighbordistances(self.pp2_str,
                                                 keep_zero_dist=False)
        self.assertAlmostEqual(nn_c[18][1], known_nonzero, places=4)
        del self.ntw.alldistances
        del self.ntw.distancematrix
        
        # nearest neighbor with snap distance
        known_neigh = ([3], 402.5219673922477)[1]
        nn_c = self.ntw.nearestneighbordistances(self.pp2_str,
                                                 keep_zero_dist=True,
                                                 snap_dist=True)
        self.assertAlmostEqual(nn_c[0][1], known_neigh, places=4)
        del self.ntw.alldistances
        del self.ntw.distancematrix
    
    @unittest.skipIf(GEOPANDAS_EXTINCT, 'Missing Geopandas')
    def test_element_as_gdf(self):
        obs = spgh.element_as_gdf(self.ntw, pp_name=self.pp1_str)
        snap_obs = spgh.element_as_gdf(self.ntw, pp_name=self.pp1_str,
                                       snapped=True)
        
        known_dist = 205.65961300587043
        observed_point = obs.loc[(obs['id']==0), 'geometry'].squeeze()
        snap_point = snap_obs.loc[(snap_obs['id']==0), 'geometry'].squeeze()
        observed_dist = observed_point.distance(snap_point)
        self.assertAlmostEqual(observed_dist, known_dist, places=8)
        
        with self.assertRaises(KeyError):
            spgh.element_as_gdf(self.ntw, pp_name='i_should_not_exist')


class TestNetworkAnalysis(unittest.TestCase):
    
    def setUp(self):
        path_to_shp = examples.get_path('streets.shp')
        self.ntw = spgh.Network(in_data=path_to_shp)
        self.pt_str = 'schools'
        path_to_shp = examples.get_path('%s.shp' % self.pt_str )
        self.ntw.snapobservations(path_to_shp, self.pt_str , attribute=True)
        npts = self.ntw.pointpatterns[self.pt_str].npoints
        self.ntw.simulate_observations(npts)
        self.test_permutations = 3
        self.test_steps = 5
    
    def tearDown(self):
        pass
    
    def test_network_f(self):
        obtained = self.ntw.NetworkF(self.ntw.pointpatterns[self.pt_str],
                                     permutations= self.test_permutations,
                                     nsteps=self.test_steps)
        self.assertEqual(obtained.lowerenvelope.shape[0], self.test_steps)
    
    def test_network_g(self):
        obtained = self.ntw.NetworkG(self.ntw.pointpatterns[self.pt_str],
                                     permutations= self.test_permutations,
                                     nsteps=self.test_steps)
        self.assertEqual(obtained.lowerenvelope.shape[0], self.test_steps)
    
    def test_network_k(self):
        obtained = self.ntw.NetworkK(self.ntw.pointpatterns[self.pt_str],
                                     permutations= self.test_permutations,
                                     nsteps=self.test_steps)
        self.assertEqual(obtained.lowerenvelope.shape[0], self.test_steps)


class TestNetworkUtils(unittest.TestCase):
    
    def setUp(self):
        path_to_shp = examples.get_path('streets.shp')
        self.ntw = spgh.Network(in_data=path_to_shp)
    
    def tearDown(self):
        pass
    
    def test_compute_length(self):
        self.point1, self.point2 = (0,0), (1,1)
        self.length = util.compute_length( self.point1, self.point2)
        self.assertAlmostEqual(self.length, 1.4142135623730951, places=4)
    
    def test_get_neighbor_distances(self):
        self.known_neighs = {1: 102.62353453439829, 2: 660.000001049743}
        self.neighs = util.get_neighbor_distances(self.ntw, 0,
                                                  self.ntw.arc_lengths)
        self.assertAlmostEqual(self.neighs[1], 102.62353453439829, places=4)
        self.assertAlmostEqual(self.neighs[2], 660.000001049743, places=4)
    
    def test_generate_tree(self):
        self.known_path = [23, 22, 20, 19, 170, 2, 0]
        self.distance, self.pred = util.dijkstra(self.ntw, 0)
        self.tree = util.generatetree(self.pred)
        self.assertEqual(self.tree[3], self.known_path)
    
    def test_dijkstra(self):
        self.distance, self.pred = util.dijkstra(self.ntw, 0)
        self.assertAlmostEqual(self.distance[196], 5505.668247, places=4)
        self.assertEqual(self.pred[196], 133)
    
    def test_dijkstra_mp(self):
        self.distance, self.pred = util.dijkstra_mp((self.ntw, 0))
        self.assertAlmostEqual(self.distance[196], 5505.668247, places=4)
        self.assertEqual(self.pred[196], 133)
    
    def test_squared_distance_point_link(self):
        self.point, self.link = (1,1), ((0,0), (2,0))
        self.sqrd_nearp = util.squared_distance_point_link(self.point,
                                                           self.link)
        self.assertEqual(self.sqrd_nearp[0], 1.0)
        self.assertEqual(self.sqrd_nearp[1].all(), np.array([1., 0.]).all())
    
    def test_snap_points_to_links(self):
        self.points = {0: cg.shapes.Point((1,1))}
        self.links = [cg.shapes.Chain([cg.shapes.Point((0,0)),
                                       cg.shapes.Point((2,0))])]
        self.snapped = util.snap_points_to_links(self.points, self.links)
        self.known_coords = [xy._Point__loc for xy in self.snapped[0][0]]
        self.assertEqual(self.known_coords, [(0.0, 0.0), (2.0, 0.0)])
        self.assertEqual(self.snapped[0][1].all(), np.array([1., 0.]).all())


if __name__ == '__main__':
    unittest.main()
