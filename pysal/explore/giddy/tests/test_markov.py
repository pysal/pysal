import unittest
import pysal.lib as ps
import numpy as np
import pysal.viz.mapclassify as mc
from ..markov import Markov, kullback, prais, Spatial_Markov, LISA_Markov

RTOL = 0.00001


class test_Markov(unittest.TestCase):
    def test___init__(self):
        # markov = Markov(class_ids, classes)
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
        m = Markov(q5)
        expected = np.array([[729., 71., 1., 0., 0.],
                             [72., 567., 80., 3., 0.],
                             [0., 81., 631., 86., 2.],
                             [0., 3., 86., 573., 56.],
                             [0., 0., 1., 57., 741.]])
        np.testing.assert_array_equal(m.transitions, expected)
        expected = np.array([[0.91011236, 0.0886392,
                               0.00124844, 0., 0.],
                              [0.09972299, 0.78531856, 0.11080332, 0.00415512,
                                  0.],
                              [0., 0.10125, 0.78875, 0.1075, 0.0025],
                              [0., 0.00417827, 0.11977716, 0.79805014,
                                  0.07799443],
                              [0., 0., 0.00125156, 0.07133917, 0.92740926]])
        np.testing.assert_array_almost_equal(m.p, expected)
        expected = np.array([0.20774716, 0.18725774, 0.20740537, 0.18821787,
                           0.20937187])
        np.testing.assert_array_almost_equal(m.steady_state, expected)


class test_Spatial_Markov(unittest.TestCase):
    def setUp(self):
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        pci = pci.transpose()
        self.rpci = pci / (pci.mean(axis=0))
        self.discretized = (self.rpci * 100).astype(int) % 4
        self.w = ps.io.open(ps.examples.get_path("states48.gal")).read()
        self.w.transform = 'r'

    def test___init__(self):
        sm = Spatial_Markov(self.rpci, self.w, fixed=True, k=5, m=5)
        S = np.array([[0.43509425, 0.2635327, 0.20363044, 0.06841983,
                       0.02932278], [0.13391287, 0.33993305, 0.25153036,
                                     0.23343016, 0.04119356], [0.12124869,
                                                               0.21137444,
                                                               0.2635101,
                                                               0.29013417,
                                                               0.1137326],
                      [0.0776413, 0.19748806, 0.25352636, 0.22480415,
                       0.24654013], [0.01776781, 0.19964349, 0.19009833,
                                     0.25524697, 0.3372434]])
        np.testing.assert_array_almost_equal(S, sm.S)

    def test_cutoff(self):
        cc = np.array([0.8, 0.9, 1, 1.2])
        sm = Spatial_Markov(self.rpci, self.w, cutoffs=cc, lag_cutoffs=cc)
        P = np.array([[[0.96703297, 0.03296703, 0.        , 0.        , 0.        ],
        [0.10638298, 0.68085106, 0.21276596, 0.        , 0.        ],
        [0.        , 0.14285714, 0.7755102 , 0.08163265, 0.        ],
        [0.        , 0.        , 0.5       , 0.5       , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]],

       [[0.88636364, 0.10606061, 0.00757576, 0.        , 0.        ],
        [0.04402516, 0.89308176, 0.06289308, 0.        , 0.        ],
        [0.        , 0.05882353, 0.8627451 , 0.07843137, 0.        ],
        [0.        , 0.        , 0.13846154, 0.86153846, 0.        ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]],

       [[0.78082192, 0.17808219, 0.02739726, 0.01369863, 0.        ],
        [0.03488372, 0.90406977, 0.05813953, 0.00290698, 0.        ],
        [0.        , 0.05919003, 0.84735202, 0.09034268, 0.00311526],
        [0.        , 0.        , 0.05811623, 0.92985972, 0.01202405],
        [0.        , 0.        , 0.        , 0.14285714, 0.85714286]],

       [[0.82692308, 0.15384615, 0.        , 0.01923077, 0.        ],
        [0.0703125 , 0.7890625 , 0.125     , 0.015625  , 0.        ],
        [0.00295858, 0.06213018, 0.82248521, 0.10946746, 0.00295858],
        [0.        , 0.00185529, 0.07606679, 0.88497217, 0.03710575],
        [0.        , 0.        , 0.        , 0.07803468, 0.92196532]],

       [[0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.06666667, 0.9       , 0.03333333, 0.        ],
        [0.        , 0.        , 0.05660377, 0.90566038, 0.03773585],
        [0.        , 0.        , 0.        , 0.03932584, 0.96067416]]])
        np.testing.assert_array_almost_equal(P, sm.P)

    def test_discretized(self):
        w = ps.weights.Queen.from_shapefile(
            ps.examples.get_path('us48.shp'))
        np.random.seed(24788)
        sm = Spatial_Markov(self.discretized, w, discrete=True)
        answer = np.array([[[  92.,   88.,   75.,   95.],
        [  50.,   55.,   52.,   35.],
        [  45.,   48.,   58.,   48.],
        [  45.,   32.,   39.,   51.]],

       [[  54.,   43.,   40.,   51.],
        [  92.,   97.,   91.,   89.],
        [  44.,   49.,   56.,   55.],
        [  40.,   35.,   75.,   50.]],

       [[  67.,   51.,   43.,   58.],
        [  41.,   58.,   56.,   35.],
        [  86.,   88.,  140.,   89.],
        [  42.,   56.,   61.,   73.]],

       [[  56.,   51.,   39.,   38.],
        [  50.,   49.,   50.,   45.],
        [  41.,   61.,   55.,   46.],
        [  93.,   77.,   87.,   89.]]])

        np.testing.assert_array_equal(sm.T, answer)
        

class test_chi2(unittest.TestCase):
    def test_chi2(self):
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        pci = pci.transpose()
        rpci = pci / (pci.mean(axis=0))
        w = ps.io.open(ps.examples.get_path("states48.gal")).read()
        w.transform = 'r'
        sm = Spatial_Markov(rpci, w, fixed=True, k=5, m=5)
        chi = np.array([[4.05598541e+01, 6.44644317e-04, 1.60000000e+01],
                         [5.54751974e+01, 2.97033748e-06, 1.60000000e+01],
                         [1.77528996e+01, 3.38563882e-01, 1.60000000e+01],
                         [4.00390961e+01, 7.68422046e-04, 1.60000000e+01],
                         [4.67966803e+01, 7.32512065e-05,
                          1.60000000e+01]])
        obs = np.array(sm.chi2)
        np.testing.assert_array_almost_equal(obs, chi)
        obs = np.array(
            [[4.61209613e+02, 0.00000000e+00, 4.00000000e+00],
             [1.48140694e+02, 0.00000000e+00, 4.00000000e+00],
             [6.33129261e+01, 5.83089133e-13, 4.00000000e+00],
             [7.22778509e+01, 7.54951657e-15, 4.00000000e+00],
             [2.32659201e+02, 0.00000000e+00, 4.00000000e+00]])
        np.testing.assert_array_almost_equal(obs, np.array(sm.shtest))


class test_LISA_Markov(unittest.TestCase):
    def test___init__(self):
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array(
            [f.by_col[str(y)] for y in range(1929, 2010)]).transpose()
        w = ps.io.open(ps.examples.get_path("states48.gal")).read()
        lm = LISA_Markov(pci, w)
        obs = np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(obs, lm.classes)
        ss = np.array([0.28561505, 0.14190226, 0.40493672, 0.16754598])
        np.testing.assert_array_almost_equal(lm.steady_state, ss)
        transitions = np.array([[1.08700000e+03, 4.40000000e+01,
                                 4.00000000e+00, 3.40000000e+01], [
                                     4.10000000e+01, 4.70000000e+02,
                                     3.60000000e+01, 1.00000000e+00], [
                                         5.00000000e+00, 3.40000000e+01,
                                         1.42200000e+03, 3.90000000e+01], [
                                             3.00000000e+01,   1.00000000e+00,
                                             4.00000000e+01, 5.52000000e+02]])
        np.testing.assert_array_almost_equal(lm.transitions, transitions)
        p = np.array([[0.92985458,  0.03763901,  0.00342173,  0.02908469],
                       [0.07481752, 0.85766423, 0.06569343, 0.00182482],
                       [0.00333333, 0.02266667, 0.948, 0.026], [0.04815409,
                                                                0.00160514,
                                                                0.06420546,
                                                                0.88603531]])
        np.testing.assert_array_almost_equal(lm.p, p)
        np.random.seed(10)
        lm_random = LISA_Markov(pci, w, permutations=99)
        expected = np.array([[1.12328098e+03,   1.15377356e+01,
                              3.47522158e-01, 3.38337644e+01], [
                                  3.50272664e+00,   5.28473882e+02,
                                  1.59178880e+01, 1.05503814e-01], [
                                      1.53878082e-01,   2.32163556e+01,
                                      1.46690710e+03, 9.72266513e+00], [
                                          9.60775143e+00,   9.86856346e-02,
                                          6.23537392e+00, 6.07058189e+02]])
        np.testing.assert_allclose(lm_random.expected_t, expected, RTOL)
        c = np.array([1058.207904, 0., 9.])
        np.testing.assert_allclose(lm_random.chi_2, c, RTOL)


class test_kullback(unittest.TestCase):
    def test___init__(self):
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
        res = kullback(F)
        np.testing.assert_array_almost_equal(160.96060031170782,
                                             res['Conditional homogeneity'])
        dof = res['Conditional homogeneity dof']
        np.testing.assert_array_almost_equal(30, dof)
        p_value = res['Conditional homogeneity pvalue']
        np.testing.assert_array_almost_equal(0.0, p_value)

class test_prais(unittest.TestCase):
    def test___init__(self):
        f = ps.io.open(ps.examples.get_path('usjoin.csv'))
        pci = np.array([f.by_col[str(y)] for y in range(1929, 2010)])
        q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
        m = Markov(q5)
        res = np.array([0.08988764, 0.21468144, 0.21125   , 0.20194986,
                        0.07259074])
        np.testing.assert_array_almost_equal(prais(m.p), res)

if __name__ == '__main__':
    unittest.main()
