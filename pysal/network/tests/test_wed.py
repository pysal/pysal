import unittest
from pysal.network.data import WED
import pysal.network.net_shp_io as net_shp_io

import pysal as ps


class TestWedOrdered(unittest.TestCase):

    def setUp(self):
        # Generate the test graph

        # from eberly http: //www.geometrictools.com/Documentation/MinimalCycleBasis.pdf
        self.coords = {0: (1, 8), 1: (1, 7), 2: (4, 7), 3: (0, 4), 4: (5, 4), 5: (3, 5),
                       6: (2, 4.5), 7: (6.5, 9), 8: (6.2, 5), 9: (5.5, 3), 10: (7, 3),
                       11: (7.5, 7.25), 12: (8, 4), 13: (11.5, 7.25), 14: (9, 1),
                       15: (11, 3), 16: (12, 2), 17: (12, 5), 18: (13.5, 6),
                       19: (14, 7.25), 20: (16, 4), 21: (18, 8.5), 22: (16, 1),
                       23: (21, 1), 24: (21, 4), 25: (18, 3.5), 26: (17, 2),
                       27: (19, 2)}
        # adjacency lists
        vertices = {}
        for v in range(28):
            vertices[v] = []

        vertices[1] = [2, 3]
        vertices[2] = [1, 4, 7]
        vertices[3] = [1, 4]
        vertices[4] = [2, 3, 5]
        vertices[5] = [4, 6]
        vertices[6] = [5]
        vertices[7] = [2, 11]
        vertices[8] = [9, 10]
        vertices[9] = [8, 10]
        vertices[10] = [8, 9]
        vertices[11] = [7, 12, 13]
        vertices[12] = [11, 13, 20]
        vertices[13] = [11, 12, 18]
        vertices[14] = [15]
        vertices[15] = [14, 16]
        vertices[16] = [15]
        vertices[18] = [13, 19]
        vertices[19] = [18, 20, 21]
        vertices[20] = [12, 19, 21, 22, 24]
        vertices[21] = [19, 20]
        vertices[22] = [20, 23]
        vertices[23] = [22, 24]
        vertices[24] = [20, 23]
        vertices[25] = [26, 27]
        vertices[26] = [25, 27]
        vertices[27] = [25, 26]

        self.edges = []
        for vert in vertices:
            for dest in vertices[vert]:
                self.edges.append((vert, dest))

    def test_wed(self):
        wed = WED(self.edges, self.coords)
        self.assertEqual(wed.enum_edges_region(0),
                        [(4, 3), (3, 1), (1, 2), (2, 4), (4, 5),
                            (5, 6), (6, 5), (5, 4), (4, 3)])
        self.assertEqual(wed.enum_links_node(20),
                        [(19, 20), (21, 20), (20, 24), (22, 20), (20, 12)])


class TestWedUnordered(unittest.TestCase):

    def setUp(self):
        self.coords = {0: (0.0, 4.0), 1: (1.0, 7.0), 2: (2.0, 4.5), 3: (3.0, 5.0), 4: (4.0, 7.0),
                       5: (5.0, 4.0), 6: (5.5, 3.0), 7: (6.2, 5.0), 8: (6.5, 9.0), 9: (7.0, 3.0),
                       10: (7.5, 7.25), 11: (8.0, 4.0), 12: (9.0, 1.0), 13: (11.0, 3.0), 14: (11.5, 7.25),
                       15: (12.0, 2.0), 16: (13.5, 6.0), 17: (14.0, 7.25), 18: (16.0, 1.0),
                       19: (16.0, 4.0), 20: (17.0, 2.0), 21: (18.0, 3.5), 22: (18.0, 8.5),
                       23: (19.0, 2.0), 24: (21.0, 1.0), 25: (21.0, 4.0)}

        self.edges = [(1, 0), (4, 1), (4, 5), (4, 8), (0, 5), (5, 3), (3, 2),
                      (8, 10), (7, 6), (7, 9), (6, 9), (10, 11), (10, 14),
                      (11, 14), (11, 19), (14, 16), (12, 13), (13, 15),
                      (16, 17), (17, 19), (17, 22), (19, 22), (19, 18),
                      (19, 25), (18, 24), (24, 25), (21, 20), (21, 23),
                      (20, 23)]

    def test_wed(self):
        wed = WED(self.edges, self.coords)
        self.assertEqual(wed.enum_edges_region(3),
                         [(19, 11), (11, 14), (14, 16), (16, 17), (17, 19), (19, 11)])
        self.assertEqual(wed.enum_links_node(19),
                         [(17, 19), (22, 19), (19, 25), (18, 19), (19, 11)])


class TestWedIOReader(unittest.TestCase):

    def setUp(self):
        self.coords, self.edges = net_shp_io.reader(ps.examples.get_path('eberly_net.shp'))

    def test_wed(self):
        wed = WED(self.edges, self.coords)
        self.assertEqual(wed.enum_edges_region(0),
                         [(5, 0), (0, 1), (1, 4), (4, 5), (5, 3), (3, 2), (2, 3), (3, 5), (5, 0)])
        self.assertEqual(wed.enum_links_node(19),
                         [(17, 19), (22, 19), (19, 25), (18, 19), (19, 11)])

        self.assertEqual(wed.w_links().histogram, [(1, 3), (2, 9), (3, 8), (4, 4), (5, 3), (6, 2)])
if __name__ == '__main__':
  unittest.main()
