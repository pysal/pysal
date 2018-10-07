
from ...common import RTOL, ATOL, pandas
from ...cg.kdtree import KDTree, RADIUS_EARTH_KM
from ..util import get_points_array
from ... import cg
from ... import weights
from .. import distance as d, contiguity as c
from ...io import geotable as pdio
from ...io.fileio import FileIO as psopen
import numpy as np
from ... import examples as pysal_examples
import unittest as ut

PANDAS_EXTINCT = pandas is None
# All instances should test these four methods, and define their own functional
# tests based on common codepaths/estimated weights use cases. 

class Distance_Mixin(object):
    polygon_path = pysal_examples.get_path('columbus.shp')
    arc_path = pysal_examples.get_path('stl_hom.shp')
    points = [(10, 10), (20, 10), (40, 10), 
              (15, 20), (30, 20), (30, 30)]
    euclidean_kdt = KDTree(points, distance_metric='euclidean')
    
    polygon_f = psopen(polygon_path) # our file handler
    poly_centroids = get_points_array(polygon_f) # our iterable
    polygon_f.seek(0) #go back to head of file
    
    arc_f = psopen(arc_path)
    arc_points = get_points_array(arc_f)
    arc_f.seek(0)
    arc_kdt = KDTree(arc_points, distance_metric='Arc',
                     radius=cg.sphere.RADIUS_EARTH_KM)
    
    cls = object # class constructor
    known_wi = None #index of known w entry to compare
    known_w = dict() #actual w entry
    known_name = known_wi
    
    def setUp(self):
        self.__dict__.update({k:v for k,v in list(Distance_Mixin.__dict__.items())
            if not k.startswith('_')})
    
    def test_init(self):
        # test vanilla, named
        raise NotImplementedError('You need to implement this test '
                                  'before this module will pass')

    def test_from_shapefile(self):
        # test vanilla, named, sparse
        raise NotImplementedError('You need to implement this test '
                                  'before this module will pass')

    def test_from_array(self):
        # test named, sparse
        raise NotImplementedError('You need to implement this test '
                                  'before this module will pass')

    def test_from_dataframe(self):
        # test named, columnar, defau
        raise NotImplementedError('You need to implement this test '
                                  'before this module will pass')

class Test_KNN(ut.TestCase, Distance_Mixin):
    def setUp(self):
        Distance_Mixin.setUp(self)
        
        self.known_wi0 = 7
        self.known_w0 = [3, 6, 12, 11]
        self.known_wi1 = 0
        self.known_w1 = [2, 1, 3 ,7]

        self.known_wi2 = 4
        self.known_w2 = [1, 3, 9, 12]
        self.known_wi3 = 40
        self.known_w3 = [31, 38, 45, 49]
    
    ##########################
    # Classmethod tests      #
    ##########################

    def test_init(self):
        w = d.KNN(self.euclidean_kdt, k=2)
        self.assertEqual(w.neighbors[0], [1,3])

    @ut.skipIf(PANDAS_EXTINCT, 'Missing pandas')
    def test_from_dataframe(self):
        df = pdio.read_files(self.polygon_path)
        w = d.KNN.from_dataframe(df, k=4)
        self.assertEqual(w.neighbors[self.known_wi0], self.known_w0)
        self.assertEqual(w.neighbors[self.known_wi1], self.known_w1)

    def test_from_array(self):
        w = d.KNN.from_array(self.poly_centroids, k=4)
        self.assertEqual(w.neighbors[self.known_wi0], self.known_w0)
        self.assertEqual(w.neighbors[self.known_wi1], self.known_w1)

    def test_from_shapefile(self):
        w = d.KNN.from_shapefile(self.polygon_path, k=4)    
        self.assertEqual(w.neighbors[self.known_wi0], self.known_w0)
        self.assertEqual(w.neighbors[self.known_wi1], self.known_w1)

    ##########################
    # Function/User tests    #
    ##########################

    def test_reweight(self):
        w = d.KNN(self.points, k=2)
        new_point = [(21,21)]
        wnew = w.reweight(k=4, p=1, new_data=new_point, inplace=False)
        self.assertEqual(wnew[0], {1: 1.0, 3: 1.0, 4: 1.0, 6: 1.0})

    def test_arcdata(self):
        w = d.KNN.from_shapefile(self.polygon_path, k=4, 
                                 distance_metric='Arc', 
                                 radius=cg.sphere.RADIUS_EARTH_KM)
        self.assertEqual(w.data.shape[1], 3)


class Test_DistanceBand(ut.TestCase, Distance_Mixin):
    def setUp(self):
        Distance_Mixin.setUp(self)
        self.grid_path =  pysal_examples.get_path('lattice10x10.shp')
        self.grid_rook_w = c.Rook.from_shapefile(self.grid_path)
        self.grid_f = psopen(self.grid_path)
        self.grid_points = get_points_array(self.grid_f)
        self.grid_f.seek(0)

        self.grid_kdt = KDTree(self.grid_points)
    
    ##########################
    # Classmethod tests      #
    ##########################

    def test_init(self):
        w = d.DistanceBand(self.grid_kdt, 1)
        for k,v in w:
            self.assertEqual(v, self.grid_rook_w[k])

    def test_from_shapefile(self):
        w = d.DistanceBand.from_shapefile(self.grid_path, 1)
        for k,v in w:
            self.assertEqual(v, self.grid_rook_w[k])

    def test_from_array(self):
        w = d.DistanceBand.from_array(self.grid_points, 1)
        for k,v in w:
            self.assertEqual(v, self.grid_rook_w[k])

    @ut.skipIf(PANDAS_EXTINCT, 'Missing pandas')
    def test_from_dataframe(self):
        import pandas as pd
        geom_series = pdio.shp.shp2series(self.grid_path)
        random_data = np.random.random(size=len(geom_series))
        df = pd.DataFrame({'obs':random_data, 'geometry':geom_series})
        w = d.DistanceBand.from_dataframe(df, 1)
        for k,v in w:
            self.assertEqual(v, self.grid_rook_w[k])

    ##########################
    # Function/User tests    #
    ##########################
    def test_integers(self):
        """
        see issue #126
        """
        grid_integers = [tuple(map(int, poly.vertices[0])) 
                              for poly in self.grid_f]
        self.grid_f.seek(0)
        grid_dbw = d.DistanceBand(grid_integers, 1)
        for k,v in grid_dbw:
            self.assertEqual(v, self.grid_rook_w[k])

    def test_arcdist(self):
        arc = cg.sphere.arcdist
        kdt = KDTree(self.arc_points, distance_metric='Arc',
                     radius=cg.sphere.RADIUS_EARTH_KM)
        npoints = self.arc_points.shape[0]
        full = np.matrix([[arc(self.arc_points[i], self.arc_points[j])
                          for j in range(npoints)] 
                          for i in range(npoints)])
        maxdist = full.max()
        w = d.DistanceBand(kdt, maxdist, binary=False, alpha=1.0)
        np.testing.assert_allclose(w.sparse.todense(), full)
        self.assertEqual(w.data.shape[1], 3)

    def test_dense(self):
        w_rook = c.Rook.from_shapefile(
                pysal_examples.get_path('lattice10x10.shp'))
        polys = psopen(pysal_examples.get_path('lattice10x10.shp'))
        centroids = [p.centroid for p in polys]
        w_db = d.DistanceBand(centroids, 1, build_sp=False)

        for k in w_db.id_order:
            np.testing.assert_equal(w_db[k], w_rook[k])
    
    @ut.skipIf(PANDAS_EXTINCT, 'Missing pandas')
    def test_named(self):
        import pandas as pd
        geom_series = pdio.shp.shp2series(self.grid_path)
        random_data = np.random.random(size=len(geom_series))
        names = [chr(x) for x in range(60,160)]
        df = pd.DataFrame({'obs':random_data, 'geometry':geom_series, 'names':names})
        w = d.DistanceBand.from_dataframe(df, 1, ids=df.names)

class Test_Kernel(ut.TestCase, Distance_Mixin):
    def setUp(self):

        Distance_Mixin.setUp(self)
        self.known_wi0 = 0
        self.known_w0 = {0: 1, 1: 0.500000049999995, 3: 0.4409830615267465}

        self.known_wi1 = 0
        self.known_w1 = {0: 1.0, 1: 0.33333333333333337,
                         3: 0.2546440075000701}
        self.known_w1_bw = 15.

        self.known_wi2 = 0
        self.known_w2 = {0: 1.0, 1: 0.59999999999999998,
                         3: 0.55278640450004202, 4: 0.10557280900008403}
        self.known_w2_bws = [25.0, 15.0, 25.0, 16.0, 14.5, 25.0]

        self.known_wi3 = 0
        self.known_w3 = [1.0, 0.10557289844279438, 9.9999990066379496e-08]
        self.known_w3_abws =[[11.180341005532938], [11.180341005532938],
                             [20.000002000000002], [11.180341005532938],
                             [14.142137037944515], [18.027758180095585]]

        self.known_wi4 = 0
        self.known_w4 = {0: 0.3989422804014327,
                         1: 0.26741902915776961,
                         3: 0.24197074871621341}
        self.known_w4_abws = self.known_w3_abws

        self.known_wi5 = 1
        self.known_w5 = {4: 0.0070787731484506233,
                         2: 0.2052478782400463,
                         3: 0.23051223027663237,
                         1: 1.0}

        self.known_wi6 = 0
        self.known_w6 = {0: 1.0, 2: 0.03178906767736345,
                         1: 9.9999990066379496e-08}
        #stick answers & params here

    ##########################
    # Classmethod tests      #
    ##########################

    def test_init(self):
        w = d.Kernel(self.euclidean_kdt)
        for k,v in list(w[self.known_wi0].items()):
            np.testing.assert_allclose(v, self.known_w0[k], rtol=RTOL)

    def test_from_shapefile(self):
        w = d.Kernel.from_shapefile(self.polygon_path, idVariable='POLYID')
        for k,v in list(w[self.known_wi5].items()):
            np.testing.assert_allclose((k,v), (k,self.known_w5[k]), rtol=RTOL)
        
        w = d.Kernel.from_shapefile(self.polygon_path, fixed=False)
        for k,v in list(w[self.known_wi6].items()):
            np.testing.assert_allclose((k,v), (k,self.known_w6[k]), rtol=RTOL)

    def test_from_array(self):
        w = d.Kernel.from_array(self.points)
        for k,v in list(w[self.known_wi0].items()):
            np.testing.assert_allclose(v, self.known_w0[k], rtol=RTOL)
    
    @ut.skipIf(PANDAS_EXTINCT, 'Missing pandas')
    def test_from_dataframe(self):
        df = pdio.read_files(self.polygon_path)
        w = d.Kernel.from_dataframe(df)
        for k,v in list(w[self.known_wi5-1].items()):
            np.testing.assert_allclose(v, self.known_w5[k+1], rtol=RTOL)
    
    ##########################
    # Function/User tests    # 
    ##########################

    def test_fixed_bandwidth(self):
        w = d.Kernel(self.points, bandwidth=15.0)
        for k,v in list(w[self.known_wi1].items()):
            np.testing.assert_allclose((k,v), (k, self.known_w1[k]))
        np.testing.assert_allclose(np.ones((w.n,1))*15, w.bandwidth)

        w = d.Kernel(self.points, bandwidth=self.known_w2_bws)
        for k,v in list(w[self.known_wi2].items()):
            np.testing.assert_allclose((k,v), (k, self.known_w2[k]), rtol=RTOL)
        for i in range(w.n):
            np.testing.assert_allclose(w.bandwidth[i], self.known_w2_bws[i], rtol=RTOL)
    
    def test_adaptive_bandwidth(self):
        w = d.Kernel(self.points, fixed=False)
        np.testing.assert_allclose(sorted(w[self.known_wi3].values()),
                                   sorted(self.known_w3), rtol=RTOL)
        bws = w.bandwidth.tolist()
        np.testing.assert_allclose(bws, self.known_w3_abws, rtol=RTOL)

        w = d.Kernel(self.points, fixed=False, function='gaussian')
        for k,v in list(w[self.known_wi4].items()):
            np.testing.assert_allclose((k,v), (k, self.known_w4[k]), rtol=RTOL)
        bws = w.bandwidth.tolist()
        np.testing.assert_allclose(bws, self.known_w4_abws, rtol=RTOL)

    def test_arcdistance(self):
        w = d.Kernel(self.points, fixed=True, distance_metric='Arc', 
                     radius=cg.sphere.RADIUS_EARTH_KM)
        self.assertEqual(w.data.shape[1], 3)

knn = ut.TestLoader().loadTestsFromTestCase(Test_KNN)
kern = ut.TestLoader().loadTestsFromTestCase(Test_Kernel)
db = ut.TestLoader().loadTestsFromTestCase(Test_DistanceBand)
suite = ut.TestSuite([knn, kern, db])
if __name__ == '__main__':
    runner = ut.TextTestRunner()
    runner.run(suite)
