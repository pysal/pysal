from ..shapes import Point, Chain, asShape
from ...io.fileio import FileIO as psopen
from ... import examples as pysal_examples
import doctest
import unittest


class test_MultiPloygon(unittest.TestCase):

    def test___init__1(self):
        """
        Tests conversion of polygons with multiple shells to 
        geoJSON multipolygons. and back.
        """
        shp = psopen(pysal_examples.get_path("NAT.shp"),'r')
        multipolygons = [p for p in shp if len(p.parts) > 1]
        geoJSON = [p.__geo_interface__ for p in multipolygons]
        for poly in multipolygons:
            json = poly.__geo_interface__
            shape = asShape(json)
            self.assertEqual(json['type'],'MultiPolygon')
            self.assertEqual(str(shape.holes), str(poly.holes))
            self.assertEqual(str(shape.parts), str(poly.parts))
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
        self.assertEqual(json['type'], 'LineString')
        self.assertEqual(len(json['coordinates']), 3)

        json = chain1.__geo_interface__
        self.assertEqual(json['type'], 'LineString')
        self.assertEqual(len(json['coordinates']), 3)

        json = chain2.__geo_interface__
        self.assertEqual(json['type'], 'MultiLineString')
        self.assertEqual(len(json['coordinates']), 2)

        chain3 = asShape(json)
        self.assertEqual(chain2.parts, chain3.parts)


if __name__ == '__main__':
    unittest.main()
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
