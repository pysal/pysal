import unittest
import pysal.lib as ps
import numpy as np
import pysal.viz.mapclassify as mc
from ..markov import Markov
from ..mobility import markov_mobility


class Shorrock_Tester(unittest.TestCase):
    def test___init__(self):
        import numpy as np
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
        m = Markov(q5)
        np.testing.assert_array_almost_equal(markov_mobility(m.p, measure="P"),
                                             0.19758992000997844)

class MarkovMobility_Tester(unittest.TestCase):
    def test___init__(self):
        pi = np.array([0.1, 0.2, 0.2, 0.4, 0.1])
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
        m = Markov(q5)
        np.testing.assert_array_almost_equal(markov_mobility(m.p, measure="P"),
                                             0.19758992000997844)
        np.testing.assert_array_almost_equal(
            markov_mobility(m.p, measure="D"),
            0.60684854623695594)
        np.testing.assert_array_almost_equal(
            markov_mobility(m.p, measure="L2"),
            0.039782002308159647)
        np.testing.assert_array_almost_equal(
            markov_mobility(m.p, measure="B1", ini=pi),
            0.2277675878319787)
        np.testing.assert_array_almost_equal(
            markov_mobility(m.p, measure="B2", ini=pi),
            0.046366601194789261)

if __name__ == '__main__':
    unittest.main()