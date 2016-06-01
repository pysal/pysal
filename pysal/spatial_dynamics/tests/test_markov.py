import unittest
import pysal
from pysal.spatial_dynamics import markov
import numpy as np


class test_Markov(unittest.TestCase):
    def test___init__(self):
        # markov = Markov(class_ids, classes)
        import pysal
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([pysal.Quantiles(y).yb for y in pci]).transpose()
        m = pysal.Markov(q5)
        expected = np.array([[729., 71., 1., 0., 0.],
                             [72., 567., 80., 3., 0.],
                             [0., 81., 631., 86., 2.],
                             [0., 3., 86., 573., 56.],
                             [0., 0., 1., 57., 741.]])
        np.testing.assert_array_equal(m.transitions, expected)
        expected = np.matrix([[0.91011236, 0.0886392,
                               0.00124844, 0., 0.],
                              [0.09972299, 0.78531856, 0.11080332, 0.00415512,
                                  0.],
                              [0., 0.10125, 0.78875, 0.1075, 0.0025],
                              [0., 0.00417827, 0.11977716, 0.79805014,
                                  0.07799443],
                              [0., 0., 0.00125156, 0.07133917, 0.92740926]])
        np.testing.assert_array_almost_equal(m.p.getA(), expected.getA())
        expected = np.matrix([[0.20774716],
                              [0.18725774],
                              [0.20740537],
                              [0.18821787],
                              [0.20937187]]).getA()
        np.testing.assert_array_almost_equal(m.steady_state.getA(), expected)


class test_Spatial_Markov(unittest.TestCase):
    def test___init__(self):
        import pysal
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        pci = pci.transpose()
        rpci = pci / (pci.mean(axis=0))
        w = pysal.open(pysal.examples.get_path("states48.gal")).read()
        w.transform = 'r'
        sm = pysal.Spatial_Markov(rpci, w, fixed=True, k=5)
        S = np.array(
            [[0.43509425, 0.2635327, 0.20363044, 0.06841983, 0.02932278],
            [0.13391287, 0.33993305, 0.25153036, 0.23343016, 0.04119356],
            [0.12124869, 0.21137444, 0.2635101, 0.29013417, 0.1137326],
            [0.0776413, 0.19748806, 0.25352636, 0.22480415, 0.24654013],
            [0.01776781, 0.19964349, 0.19009833, 0.25524697, 0.3372434]])
        np.testing.assert_array_almost_equal(S, sm.S)


class test_chi2(unittest.TestCase):
    def test_chi2(self):
        import pysal
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        pci = pci.transpose()
        rpci = pci / (pci.mean(axis=0))
        w = pysal.open(pysal.examples.get_path("states48.gal")).read()
        w.transform = 'r'
        sm = pysal.Spatial_Markov(rpci, w, fixed=True, k=5)
        chi = np.matrix([[4.05598541e+01, 6.44644317e-04, 1.60000000e+01],
		         [5.54751974e+01, 2.97033748e-06, 1.60000000e+01],
			 [1.77528996e+01, 3.38563882e-01, 1.60000000e+01],
			 [4.00390961e+01, 7.68422046e-04, 1.60000000e+01],
			 [4.67966803e+01, 7.32512065e-05,
				 1.60000000e+01]]).getA()
        obs = np.matrix(sm.chi2).getA()
        np.testing.assert_array_almost_equal(obs, chi)
        obs = np.matrix(
            [[4.61209613e+02, 0.00000000e+00, 4.00000000e+00],
             [1.48140694e+02, 0.00000000e+00, 4.00000000e+00],
             [6.33129261e+01, 5.83089133e-13, 4.00000000e+00],
             [7.22778509e+01, 7.54951657e-15, 4.00000000e+00],
             [2.32659201e+02, 0.00000000e+00, 4.00000000e+00]])
        np.testing.assert_array_almost_equal(obs.getA(),
                                             np.matrix(sm.shtest).getA())


class test_LISA_Markov(unittest.TestCase):
    def test___init__(self):
        import numpy as np
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array(
            [f.by_col[str(y)] for y in range(1929, 2010)]).transpose()
        w = pysal.open(pysal.examples.get_path("states48.gal")).read()
        lm = pysal.LISA_Markov(pci, w)
        obs = np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(obs, lm.classes)
        """
        >>> lm.steady_state
        matrix([[ 0.28561505],
                [ 0.14190226],
                [ 0.40493672],
                [ 0.16754598]])
        >>> lm.transitions
        array([[  1.08700000e+03,   4.40000000e+01,   4.00000000e+00,
                  3.40000000e+01],
               [  4.10000000e+01,   4.70000000e+02,   3.60000000e+01,
                  1.00000000e+00],
               [  5.00000000e+00,   3.40000000e+01,   1.42200000e+03,
                  3.90000000e+01],
               [  3.00000000e+01,   1.00000000e+00,   4.00000000e+01,
                  5.52000000e+02]])
        >>> lm.p
        matrix([[ 0.92985458,  0.03763901,  0.00342173,  0.02908469],
                [ 0.07481752,  0.85766423,  0.06569343,  0.00182482],
                [ 0.00333333,  0.02266667,  0.948     ,  0.026     ],
                [ 0.04815409,  0.00160514,  0.06420546,  0.88603531]])
        >>> lm.move_types
        array([[ 11.,  11.,  11., ...,  11.,  11.,  11.],
               [  6.,   6.,   6., ...,   6.,   7.,  11.],
               [ 11.,  11.,  11., ...,  11.,  11.,  11.],
               ...,
               [  6.,   6.,   6., ...,   6.,   6.,   6.],
               [  1.,   1.,   1., ...,   6.,   6.,   6.],
               [ 16.,  16.,  16., ...,  16.,  16.,  16.]])
        >>> np.random.seed(10)
        >>> lm_random = pysal.LISA_Markov(pci, w, permutations=99)
        >>> lm_random.significant_moves
        array([[11, 11, 11, ..., 59, 59, 59],
               [54, 54, 54, ..., 54, 55, 59],
               [11, 11, 11, ..., 11, 59, 59],
               ...,
               [54, 54, 54, ..., 54, 54, 54],
               [49, 49, 49, ..., 54, 54, 54],
               [64, 64, 64, ..., 64, 64, 64]])

        """


class test_kullback(unittest.TestCase):
    def test___init__(self):
        import numpy as np
        s1 = np.array([
                      [22, 11, 24, 2, 2, 7],
                      [5, 23, 15, 3, 42, 6],
                      [4, 21, 190, 25, 20, 34],
                      [0, 2, 14, 56, 14, 28],
                      [32, 15, 20, 10, 56, 14],
                      [5, 22, 31, 18, 13, 134]
                      ])
        s2 = np.array([
            [3, 6, 9, 3, 0, 8],
            [1, 9, 3, 12, 27, 5],
            [2, 9, 208, 32, 5, 18],
            [0, 14, 32, 108, 40, 40],
            [22, 14, 9, 26, 224, 14],
            [1, 5, 13, 53, 13, 116]
        ])

        F = np.array([s1, s2])
        res = markov.kullback(F)
        np.testing.assert_array_almost_equal(160.96060031170782,
                                             res['Conditional homogeneity'])
        np.testing.assert_array_almost_equal(30,
                                             res['Conditional homogeneity dof'])
        np.testing.assert_array_almost_equal(0.0,
                                             res['Conditional homogeneity pvalue'])


class test_prais(unittest.TestCase):
    def test___init__(self):
        import numpy as np
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([pysal.Quantiles(y).yb for y in pci]).transpose()
        m = pysal.Markov(q5)
        res = np.matrix([[0.08988764, 0.21468144,
                          0.21125, 0.20194986, 0.07259074]])
        np.testing.assert_array_almost_equal(markov.prais(m.p), res)


class test_shorrock(unittest.TestCase):
    def test___init__(self):
        import numpy as np
        f = pysal.open(pysal.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([pysal.Quantiles(y).yb for y in pci]).transpose()
        m = pysal.Markov(q5)
        np.testing.assert_array_almost_equal(markov.shorrock(m.p),
                                             0.19758992000997844)


if __name__ == '__main__':
    unittest.main()
