from ..voronoi import voronoi, voronoi_frames
from ..shapes import Polygon, asShape
import unittest


class Voronoi(unittest.TestCase):

    def setUp(self):
        self.points  = [(10.2, 5.1), (4.7, 2.2), (5.3, 5.7), (2.7, 5.3)]

        self.vertices = [[4.21783295711061, 4.084085778781038], [7.519560251284979, 3.518075385494004], [9.464219298524961, 19.399457604620512], [14.982106844470032, -10.63503022227075], [-9.226913414477298, -4.58994413837245], [14.982106844470032, -10.63503022227075], [1.7849180090475505, 19.898032941190912], [9.464219298524961, 19.399457604620512], [1.7849180090475505, 19.898032941190912], [-9.226913414477298, -4.58994413837245]]

    def test_voronoi(self):
        regions, vertices = voronoi(self.points)
        self.assertEqual(regions, [[1, 3, 2],
                                    [4, 5, 1, 0],
                                    [0, 1, 7, 6],
                                    [9, 0, 8]])

        self.assertTrue(vertices.tolist() == self.vertices)

    def test_voronoi_frames(self):
        r_df, p_df = voronoi_frames(self.points)
        region = r_df.iloc[0]['geometry']
        try:
            import geopandas as df
            self.assertTrue(isinstance(asShape(region), Polygon))
        except ImportError:
            self.assertTrue(isinstance(region, Polygon))



if __name__ == '__main__':
    unittest.main()
