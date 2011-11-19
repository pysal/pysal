import unittest
import numpy as np
import pysal
from pysal.spreg import ols as OLS

class Test_OLS(unittest.TestCase):
    """ setUp is called before each test function execution """
    def setUp(self):
        db=pysal.open("pysal/examples/columbus.dbf","r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        X = np.array(X).T
        self.y = y
        self.X = X

    """ All method names that begin with 'test' will be executed as a test case """
    def test_BaseOLS(self):
        ols = OLS.BaseOLS(self.y, self.X)
        # test the typical usage
        x = np.hstack((np.ones(self.y.shape),self.X))
        np.testing.assert_array_equal(ols.x, x)
        np.testing.assert_array_equal(ols.y, self.y)
        betas = np.array([[ 68.6189611 ], [ -1.59731083], [ -0.27393148]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.88337500e+03],
                        [  7.04371999e+02,   1.16866734e+04,   2.96004418e+04],
                        [  1.88337500e+03,   2.96004418e+04,   8.87576189e+04]])
        np.testing.assert_array_almost_equal(ols.xtx, xtx, decimal=4)
        xtxi = np.array([[  1.71498009e-01,  -7.20680548e-03,  -1.23561717e-03],
                         [ -7.20680548e-03,   8.53813203e-04,  -1.31821143e-04],
                         [ -1.23561717e-03,  -1.31821143e-04,   8.14475943e-05]])
        np.testing.assert_array_almost_equal(ols.xtxi, xtxi, decimal=8)
        u_sub = np.array([[  0.34654188], [ -3.694799  ], [ -5.28739398],
                       [-19.98551514], [  6.44754899]])
        np.testing.assert_array_almost_equal(ols.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[ 15.37943812], [ 22.496553  ], [ 35.91417498],
                          [ 52.37327514], [ 44.28396101]])
        np.testing.assert_array_almost_equal(ols.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(ols.n, 49)
        self.assertEquals(ols.k, 3)
        self.assertAlmostEquals(ols.utu, 6014.892735784364, places=10)
        self.assertAlmostEquals(ols.sig2n, 122.75291297519111, places=10)
        self.assertAlmostEquals(ols.sig2n_k, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        vm = np.array([[  2.24248289e+01,  -9.42351346e-01,  -1.61567494e-01],
                       [ -9.42351346e-01,   1.11643366e-01,  -1.72367399e-02],
                       [ -1.61567494e-01,  -1.72367399e-02,   1.06499683e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm, decimal=6)
        self.assertAlmostEquals(ols.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(ols.std_y, 16.732092091229699, places=10)

    def test_OLS(self):
        ols = OLS.OLS(self.y, self.X)
        x = np.hstack((np.ones(self.y.shape), self.X))
        # make sure OLS matches BaseOLS
        np.testing.assert_array_equal(ols.x, x)
        np.testing.assert_array_equal(ols.y, self.y)
        betas = np.array([[ 68.6189611 ], [ -1.59731083], [ -0.27393148]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)
        xtx = np.array([[  4.90000000e+01,   7.04371999e+02,   1.88337500e+03],
                        [  7.04371999e+02,   1.16866734e+04,   2.96004418e+04],
                        [  1.88337500e+03,   2.96004418e+04,   8.87576189e+04]])
        np.testing.assert_array_almost_equal(ols.xtx, xtx, decimal=4)
        xtxi = np.array([[  1.71498009e-01,  -7.20680548e-03,  -1.23561717e-03],
                         [ -7.20680548e-03,   8.53813203e-04,  -1.31821143e-04],
                         [ -1.23561717e-03,  -1.31821143e-04,   8.14475943e-05]])
        np.testing.assert_array_almost_equal(ols.xtxi, xtxi, decimal=8)
        u_sub = np.array([[  0.34654188], [ -3.694799  ], [ -5.28739398],
                       [-19.98551514], [  6.44754899]])
        np.testing.assert_array_almost_equal(ols.u[0:5], u_sub, decimal=8)
        predy_sub = np.array([[ 15.37943812], [ 22.496553  ], [ 35.91417498],
                          [ 52.37327514], [ 44.28396101]])
        np.testing.assert_array_almost_equal(ols.predy[0:5], predy_sub, decimal=8)
        self.assertEquals(ols.n, 49)
        self.assertEquals(ols.k, 3)
        self.assertAlmostEquals(ols.utu, 6014.892735784364, places=10)
        self.assertAlmostEquals(ols.sig2n, 122.75291297519111, places=10)
        self.assertAlmostEquals(ols.sig2n_k, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.sigML, 122.75291297519111, places=10)
        vm = np.array([[  2.24248289e+01,  -9.42351346e-01,  -1.61567494e-01],
                       [ -9.42351346e-01,   1.11643366e-01,  -1.72367399e-02],
                       [ -1.61567494e-01,  -1.72367399e-02,   1.06499683e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm, decimal=6)
        self.assertAlmostEquals(ols.mean_y, 35.128823897959187, places=10)
        self.assertAlmostEquals(ols.std_y, 16.732092091229699, places=10)
        # start testing specific attributes for the OLS class
        self.assertEquals(ols.name_ds, 'unknown')
        self.assertEquals(ols.name_x, ['CONSTANT', 'var_1', 'var_2'])
        self.assertEquals(ols.name_y, 'dep_var')
        ols = OLS.OLS(self.y, self.X, name_ds='Columbus',
                      name_x=['inc','hoval'], name_y='crime')
        self.assertEquals(ols.name_ds, 'Columbus')
        self.assertEquals(ols.name_x, ['CONSTANT', 'inc', 'hoval'])
        self.assertEquals(ols.name_y, 'crime')
        self.assertAlmostEquals(ols.r2, 0.55240404083742334, places=10)
        self.assertAlmostEquals(ols.ar2, 0.5329433469607896, places=10)
        self.assertAlmostEquals(ols.sig2, 130.75853773444268, places=10)
        self.assertAlmostEquals(ols.f_stat[0], 28.385629224694853, places=10)
        self.assertAlmostEquals(ols.f_stat[1], 9.3407471005108332e-09, places=10)
        self.assertAlmostEquals(ols.logll, -187.3772388121491, places=10)
        self.assertAlmostEquals(ols.aic, 380.7544776242982, places=10)
        self.assertAlmostEquals(ols.schwarz, 386.42993851863008, places=10)
        std_err = np.array([ 4.73548613,  0.33413076,  0.10319868])
        np.testing.assert_array_almost_equal(ols.std_err, std_err, decimal=8)
        t_stat = [(14.490373143689094, 9.2108899889173982e-19),
                  (-4.7804961912965762, 1.8289595070843232e-05),
                  (-2.6544086427176916, 0.010874504909754612)]
        np.testing.assert_array_almost_equal(ols.t_stat, t_stat, decimal=8)
        self.assertAlmostEquals(ols.mulColli, 6.5418277514438046, places=10)
        self.assertAlmostEquals(ols.jarque_bera['jb'], 1.835752520075947)
        self.assertEquals(ols.jarque_bera['df'], 2)
        self.assertAlmostEquals(ols.jarque_bera['pvalue'], 0.39936629124876566)
        self.assertAlmostEquals(ols.breusch_pagan['bp'], 7.900441675960)
        self.assertEquals(ols.breusch_pagan['df'], 2)
        self.assertAlmostEquals(ols.breusch_pagan['pvalue'], 0.019250450075)
        self.assertAlmostEquals(ols.koenker_bassett['kb'], 5.694087931707)
        self.assertEquals(ols.koenker_bassett['df'], 2)
        self.assertAlmostEquals(ols.koenker_bassett['pvalue'], 0.058015563638)
        self.assertAlmostEquals(ols.white['wh'], 19.946008239903257)
        self.assertEquals(ols.white['df'], 5)
        self.assertAlmostEquals(ols.white['pvalue'], 0.0012792228173925788)
        # test not using constant
        ols = OLS.OLS(self.y, self.X, constant=False)
        betas = np.array([[ 1.28624161], [ 0.22045774]])
        np.testing.assert_array_almost_equal(ols.betas, betas, decimal=8)
        # test spatial diagnostics
        w = pysal.open('pysal/examples/columbus.gal', 'r').read()
        w.transform = 'r'
        ols = OLS.OLS(self.y, self.X, w=w)
        self.assertAlmostEquals(ols.lm_error[0], 5.2062139238820784)
        self.assertAlmostEquals(ols.lm_error[1], 0.022506293821436953)
        self.assertAlmostEquals(ols.lm_lag[0],8.897998591087477)
        self.assertAlmostEquals(ols.lm_lag[1], 0.0028548339507328928)
        self.assertAlmostEquals(ols.rlm_error[0],0.043905931885077722)
        self.assertAlmostEquals(ols.rlm_error[1], 0.83402872393126437)
        self.assertAlmostEquals(ols.rlm_lag[0],3.7356905990904767)
        self.assertAlmostEquals(ols.rlm_lag[1], 0.053261645050770842)
        self.assertAlmostEquals(ols.lm_sarma[0], 8.9419045229725551)
        self.assertAlmostEquals(ols.lm_sarma[1], 0.011436420201077028)
        self.assertAlmostEquals(ols.moran_res[0], 0.22210940657867592)
        self.assertAlmostEquals(ols.moran_res[1], 2.8393189345136847)
        self.assertAlmostEquals(ols.moran_res[2], 0.0045209944743344263)


suite = unittest.TestLoader().loadTestsFromTestCase(Test_OLS)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
