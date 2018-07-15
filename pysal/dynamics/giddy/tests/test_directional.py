import unittest
import pysal.lib
import pysal.dynamics.giddy.directional as directional
import numpy as np


class Rose_Tester(unittest.TestCase):
    def setUp(self):
        f = open(pysal.lib.examples.get_path("spi_download.csv"), 'r')
        lines = f.readlines()
        f.close()
        lines = [line.strip().split(",") for line in lines]
        names = [line[2] for line in lines[1:-5]]
        data = np.array([list(map(int, line[3:])) for line in lines[1:-5]])
        sids = list(range(60))
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
        years = np.arange(1969, 2009)
        rel = states / (us * 1.)
        gal = pysal.lib.open(pysal.lib.examples.get_path('states48.gal'))
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


    def test_plot(self):
        import geopandas as gpd
        import pandas as pd
        import pysal.lib.api as lp
        from pysal.lib import examples
        import numpy as np
        import matplotlib.pyplot as plt
        from pysal.dynamics.giddy.directional import Rose
        # get data, calc mean, merge
        shp_link = examples.get_path('us48.shp')
        df = gpd.read_file(shp_link)
        income_table = pd.read_csv(examples.get_path("usjoin.csv"))
        for year in range(1969, 2010):
            income_table[str(year) + '_rel'] = (
                income_table[str(year)] / income_table[str(year)].mean())
        gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')
        # statistical analysis
        w = lp.Queen.from_dataframe(gdf)
        w.transform = 'r'
        y1 = gdf['1969_rel'].values
        y2 = gdf['2000_rel'].values
        Y = np.array([y1, y2]).T
        rose = Rose(Y, w, k=5)
        # plot
        fig, _ = rose.plot()
        plt.close(fig)
        # plot with atribute coloring
        fig, _ = rose.plot(attribute=y1)
        plt.close(fig)


    def test_plot_vectors(self):
        import geopandas as gpd
        import pandas as pd
        import pysal.lib.api as lp
        from pysal.lib import examples
        import numpy as np
        import matplotlib.pyplot as plt
        from pysal.dynamics.giddy.directional import Rose
        # get data, calc mean, merge
        shp_link = examples.get_path('us48.shp')
        df = gpd.read_file(shp_link)
        income_table = pd.read_csv(examples.get_path("usjoin.csv"))
        for year in range(1969, 2010):
            income_table[str(year) + '_rel'] = (
                income_table[str(year)] / income_table[str(year)].mean())
        gdf = df.merge(income_table,left_on='STATE_NAME',right_on='Name')
        # statistical analysis
        w = lp.Queen.from_dataframe(gdf)
        w.transform = 'r'
        y1 = gdf['1969_rel'].values
        y2 = gdf['2000_rel'].values
        Y = np.array([y1, y2]).T
        rose = Rose(Y, w, k=5)
        # plot
        fig, _ = rose.plot_vectors()
        plt.close(fig)
        # customize plot
        fig, _ = rose.plot_vectors(arrows=False)
        plt.close(fig)

suite = unittest.TestSuite()
test_classes = [Rose_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
