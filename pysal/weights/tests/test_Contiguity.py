from pysal.weights import Contiguity as c
import pysal as ps
import unittest as ut
from warnings import warn as Warn

PANDAS_EXTINCT = ps.common.pandas is None
class Mock(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Contiguity_Mixin(object):
    polygon_path = ps.examples.get_path('columbus.shp')
    point_path = ps.examples.get_path('baltim.shp')
    f = ps.open(polygon_path) # our file handler
    polygons = f.read() # our iterable
    f.seek(0) #go back to head of file
    # Without requiring users to have GDAL to run tests, 
    # we can mock a shapely object by an object with a geo_interface
    mocks = [Mock(__geo_interface__=p.__geo_interface__) 
                  for p in polygons] 
    cls = object # class constructor
    known_wi = None #index of known w entry to compare
    known_w = dict() #actual w entry
    known_name = known_wi
    known_namedw = known_w
    idVariable = None # id variable from file or column

    def setUp(self):
        self.__dict__.update({k:v for k,v in Contiguity_Mixin.__dict__.items()
            if not k.startswith('_')})
    
    def runTest(self):
        pass 

    def test_init(self):
        # basic
        w = self.cls(self.polygons)
        self.assertEqual(w[self.known_wi], self.known_w)

        # sparse
        #w = self.cls(self.polygons, sparse=True)
        #srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        #this_w = {i:k for i,k in enumerate(srowvec) if k>0}
        #self.assertEqual(this_w, self.known_w)
        #ids = ps.weights2.utils.get_ids(self.polygon_path, self.idVariable)

        # named
        ids = ps.weights.util.get_ids(self.polygon_path, self.idVariable)
        w = self.cls(self.polygons, ids = ids)
        self.assertEqual(w[self.known_name], self.known_namedw)

    def test_from_iterable(self):
        w = self.cls.from_iterable(self.f)
        self.f.seek(0)
        self.assertEqual(w[self.known_wi], self.known_w)
        
        w = self.cls.from_iterable(self.mocks)
        self.assertEqual(w[self.known_wi], self.known_w)

    def test_from_shapefile(self):
        # basic
        w = self.cls.from_shapefile(self.polygon_path)
        self.assertEqual(w[self.known_wi], self.known_w)

        # sparse
        ws = self.cls.from_shapefile(self.polygon_path, sparse=True)
        srowvec = ws.sparse[self.known_wi].todense().tolist()[0]
        this_w = {i:k for i,k in enumerate(srowvec) if k>0}
        self.assertEqual(this_w, self.known_w)

        # named
        w = self.cls.from_shapefile(self.polygon_path, idVariable=self.idVariable)
        self.assertEqual(w[self.known_name], self.known_namedw)

    def test_from_array(self):
        # test named, sparse from point array
        pass 

    @ut.skipIf(PANDAS_EXTINCT, 'Missing pandas')
    def test_from_dataframe(self):
        # basic
        df = ps.pdio.read_files(self.polygon_path)
        w = self.cls.from_dataframe(df)
        self.assertEqual(w[self.known_wi], self.known_w)

        # named geometry
        df.rename(columns={'geometry':'the_geom'}, inplace=True)
        w = self.cls.from_dataframe(df, geom_col = 'the_geom')
        self.assertEqual(w[self.known_wi], self.known_w)

        # named geometry + named obs
        w = self.cls.from_dataframe(df, geom_col='the_geom', idVariable=self.idVariable)
        self.assertEqual(w[self.known_name], self.known_namedw)

class Test_Queen(ut.TestCase, Contiguity_Mixin):
    def setUp(self):
        Contiguity_Mixin.setUp(self)
        
        self.known_wi = 4
        self.known_w  = {2: 1.0, 3: 1.0, 5: 1.0, 7: 1.0,
                             8: 1.0, 10: 1.0, 14: 1.0, 15: 1.0}
        self.cls = c.Queen
        self.idVariable = 'POLYID'
        self.known_name = 5
        self.known_namedw = {k+1:v for k,v in self.known_w.items()}

class Test_Rook(ut.TestCase, Contiguity_Mixin):
    def setUp(self):
        Contiguity_Mixin.setUp(self)
        
        self.known_w = {2: 1.0, 3: 1.0, 5: 1.0, 7: 1.0, 
                             8: 1.0, 10: 1.0, 14: 1.0}
        self.known_wi = 4
        self.cls = c.Rook
        self.idVariable = 'POLYID'
        self.known_name = 5
        self.known_namedw = {k+1:v for k,v in self.known_w.items()}

q = ut.TestLoader().loadTestsFromTestCase(Test_Queen)
r = ut.TestLoader().loadTestsFromTestCase(Test_Rook)
suite = ut.TestSuite([q, r])
if __name__ == '__main__':
    runner = ut.TextTestRunner()
    runner.run(suite)
