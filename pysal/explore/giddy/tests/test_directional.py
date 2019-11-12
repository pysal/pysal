import unittest
import pysal.lib as ps
from .. import directional
import numpy as np


@unittest.skip("skipping")
class Rose_Tester(unittest.TestCase):
    def setUp(self):
        f = open(ps.examples.get_path("spi_download.csv"), 'r')
        lines = f.readlines()
        f.close()
        lines = [line.strip().split(",") for line in lines]
        names = [line[2] for line in lines[1:-5]]
        data = np.array([list(map(int, line[3:])) for line in lines[1:-5]])
        out = ['"United States 3/"',
               '"Alaska 3/"',
               '"District of Columbia"',
               '"Hawaii 3/"',
               '"New England"',
               '"Mideast"',
               '"Great Lakes"',
               '"Plains"',
               '"Southeast"',
               '"Southwest"',
               '"Rocky Mountain"',
               '"Far West 3/"']
        snames = [name for name in names if name not in out]
        sids = [names.index(name) for name in snames]
        states = data[sids, :]
        us = data[0]
        rel = states / (us * 1.)
        gal = ps.io.open(ps.examples.get_path('states48.gal'))
        self.w = gal.read()
        self.w.transform = 'r'
        self.Y = rel[:, [0, -1]]

    def test_rose(self):
        k = 4
        np.random.seed(100)
        r4 = directional.Rose(self.Y, self.w, k)
        exp = [0., 1.57079633, 3.14159265, 4.71238898, 6.28318531]
        obs = list(r4.cuts)
        for i in range(k + 1):
            self.assertAlmostEqual(exp[i], obs[i])
        self.assertEqual(list(r4.counts), [32, 5, 9, 2])

        import matplotlib.pyplot as plt
        # plot
        fig, _ = r4.plot()
        plt.close(fig)
        # plot with attribute coloring
        fig, _ = r4.plot(attribute=self.Y[:,0])
        plt.close(fig)

        # plot
        fig, _ = r4.plot_vectors()
        plt.close(fig)
        # customize plot
        fig, _ = r4.plot_vectors(arrows=False)
        plt.close(fig)

suite = unittest.TestSuite()
test_classes = [Rose_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
