import pysal
import doctest
import unittest


class test_MultiPloygon(unittest.TestCase):

    def test___init__1(self):
        """
        Tests conversion of polygons with multiple shells to 
        geoJSON multipolygons. and back.
        """
        shp = pysal.open(pysal.examples.get_path("NAT.SHP"),'r')
        multipolygons = [p for p in shp if len(p.parts) > 1]
        geoJSON = [p.__geo_interface__ for p in multipolygons]
        for poly in multipolygons:
            json = poly.__geo_interface__
            shape = pysal.cg.asShape(json)
            self.assertEquals(json['type'],'MultiPolygon')
            self.assertEquals(str(shape.holes), str(poly.holes))
            self.assertEquals(str(shape.parts), str(poly.parts))

if __name__ == '__main__':
    unittest.main()
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
