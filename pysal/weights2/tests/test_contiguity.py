from pysal.weights2 import contiguity as c
import pysal as ps
import unittest as ut

class Any_Contiguity(ut.TestCase):
    def setUp(self):
        self.polygon_path = ps.examples.get_path('columbus.shp')
        self.point_path = ps.examples.get_path('baltim.shp')

    def test_init(self):
        # test from naive collection with name, sparse, point, & poly
        raise NotImplementedError

    def test_from_shapefile(self):
        # test vanilla, named, sparse
        # from point & poly shapefiles
        raise NotImplementedError

    def test_from_array(self):
        # test named, sparse from point array
        raise NotImplementedError

    def test_from_dataframe(self):
        # test named, columnar, default index
        raise NotImplementedError

class Test_Queen(Any_Contiguity):
    def setUp(self):
        Any_Contiguity.__init__(self)
        #stick answers and params here
    def test_higher_order(self):
        #test construction of higher-order queen weights
        raise NotImplementedError

class Test_Rook(Any_Contiguity):
    def setUp(self):
        Any_Contiguity.__init__(self)
        #stick answers & params here
    def test_higher_order(self):
        #test construction of higher-order rook weights
        raise NotImplementedError
