"""network unittest"""
import unittest
import network as pynet
import kernel as pykernel

class Kernel_Tester(unittest.TestCase):

    def setUp(self):
        self.G = {(1,1): {(2,2): 0.125, (3,3): 0.75}, (2,2): {(1,1): 0.125, (4,4): 1.2},
                   (3,3): {(1,1): 0.75, (4,4): 0.375},
                   (4,4): {(2,2): 1.2, (3,3): 0.375, (5,5): 0.5},
                   (5,5): {(4,4): 0.5}, (6,6):{(7,7):1.0}, (7,7):{(6,6):1.0}}
        self.G_meshed = pynet.mesh_network(self.G, 0.1)
        self.points = [((3.6666666666666665, 3.6666666666666665), (3.5, 3.5), 
                         1.8011569244041523e-18, 8.0119209423694433e-18), 
                       ((4.0, 4.0), (3.8333333333333335, 3.8333333333333335), 
                         6.4354219496947843e-18, 1.3190733783852405e-17), 
                       ((6.5999999999999996, 6.5999999999999996), (6.5, 6.5), 
                         8.6525456558003033e-19, 4.0412843678067672e-18)]
        self.proj_points = []
        for p in self.points:
            self.proj_points.append(pynet.proj_pnt_coor(p))

    def test_dijkstras_w_prev(self):
        distances, previous_nodes = pykernel.dijkstras_w_prev(self.G, (1,1))
        distances_values = {(5, 5): 1.625, (3, 3): 0.75, (4, 4): 1.125, (1, 1): 0, (2, 2): 0.125}
        prev_nodes = {(5, 5): (4, 4), (2, 2): (1, 1), (1, 1): None, (4, 4): (3, 3), (3, 3): (1, 1)}
        self.assertEqual(distances, distances_values)
        self.assertEqual(previous_nodes, prev_nodes)

    def test_kernel_density(self):
        density = pykernel.kernel_density(self.G_meshed, self.proj_points, 0.3, self.G.keys())
        self.assertEqual(density[(4.0, 4.0)], 0.25)


suite = unittest.TestSuite()
test_classes = [Kernel_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
