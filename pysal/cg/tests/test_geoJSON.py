import pysal
from pysal.cg.shapes import Point, Chain
import doctest
import unittest


class test_MultiPloygon(unittest.TestCase):

    def test___init__1(self):
        """
        Tests conversion of polygons with multiple shells to 
        geoJSON multipolygons. and back.
        """
        shp = pysal.open(pysal.examples.get_path("NAT.shp"),'r')
        multipolygons = [p for p in shp if len(p.parts) > 1]
        geoJSON = [p.__geo_interface__ for p in multipolygons]
        for poly in multipolygons:
            json = poly.__geo_interface__
            shape = pysal.cg.asShape(json)
            self.assertEquals(json['type'],'MultiPolygon')
            self.assertEquals(str(shape.holes), str(poly.holes))
            self.assertEquals(str(shape.parts), str(poly.parts))
class test_MultiLineString(unittest.TestCase):
    def test_multipart_chain(self):
        vertices = [[Point((0, 0)), Point((1, 0)), Point((1, 5))],
                    [Point((-5, -5)), Point((-5, 0)), Point((0, 0))]]

        #part A
        chain0 = Chain(vertices[0])
        #part B
        chain1 = Chain(vertices[1])
        #part A and B
        chain2 = Chain(vertices)

        json = chain0.__geo_interface__
        self.assertEquals(json['type'], 'LineString')
        self.assertEquals(len(json['coordinates']), 3)

        json = chain1.__geo_interface__
        self.assertEquals(json['type'], 'LineString')
        self.assertEquals(len(json['coordinates']), 3)

        json = chain2.__geo_interface__
        self.assertEquals(json['type'], 'MultiLineString')
        self.assertEquals(len(json['coordinates']), 2)

        chain3 = pysal.cg.asShape(json)
        self.assertEquals(chain2.parts, chain3.parts)


if __name__ == '__main__':
    unittest.main()
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
