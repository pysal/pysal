from pysal.weights2 import distance as d
import pysal as ps
import unittest as ut

# All instances should test these four methods, and define their own functional
# tests based on common codepaths/estimated weights use cases. 

class Any_Distance(ut.TestCase):
    def setUp(self):
        self.polygon_path = ps.examples.get_path('columbus.shp')
        self.point_path = ps.examples.get_path('baltim.shp')
    
    def test_init(self):
        raise NotImplementedError

    def test_from_shapefile(self):
        # test vanilla, named, sparse
        raise NotImplementedError

    def test_from_array(self):
        # test named, sparse
        raise NotImplementedError

    def test_from_dataframe(self):
        # test named, columnar, default index
        raise NotImplementedError

class Test_KNN(Any_Distance):
    def setUp(self):
        Any_Distance.__init__(self)
        #stick answers & params here

    def test_reweight(self):
        raise NotImplementedError

class Test_DistanceBand(Any_Distance):
    def setUp(self):
        Any_Distance.__init__(self)
        #stick answers & params here

    def test_Threshold_Continuous(self):
        #test common codepaths
        raise NotImplementedError

    def test_Threshold_Binary(self):
        #test common codepaths
        raise NotImplementedError

class Test_Kernel(Any_Distance):
    def setUp(self):
        Any_Distance.__init__(self)
        #stick answers & params here

    def test_Adaptive(self):
        raise NotImplementedError

    def test_Global(self):
        raise NotImplementedError
