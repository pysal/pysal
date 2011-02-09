import unittest
import numpy as np
import pysal
from pysal.spreg import ols as OLS
from pysal.spreg import user_output as USER

class Test_OLS(unittest.TestCase):
    """ setUp is called before each test function execution """
    def setUp(self):
        db=pysal.open("../../examples/columbus.dbf","r")
        y = np.array(db.by_col("CRIME"))
        y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        X = np.array(X).T
        self.y = y
        self.X = X

    def test_DiagnosticsBuilder(self):
        ols = OLS.OLS(self.y, self.X)
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
        # test spatial diagnostics
        w = pysal.open('../../examples/columbus.gal', 'r').read()
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

    def test_set_name_ds(self):
        output = USER.set_name_ds(None)
        self.assertEquals(output, 'unknown')
        output = USER.set_name_ds('my_data')
        self.assertEquals(output, 'my_data')
    def test_set_name_y(self):
        output = USER.set_name_y(None)
        self.assertEquals(output, 'dep_var')
        output = USER.set_name_y('my_variable')
        self.assertEquals(output, 'my_variable')
    def test_set_name_x(self):
        output = USER.set_name_x([], self.X, True)
        self.assertEquals(output[0], 'CONSTANT')
        self.assertEquals(output[1], 'var_1')
        self.assertEquals(output[2], 'var_2')
        output = USER.set_name_x(None, self.X, True)
        self.assertEquals(output[1], 'var_1')
        self.assertEquals(output[2], 'var_2')
        output = USER.set_name_x(None, self.X, False)
        self.assertEquals(output[0], 'var_1')
        self.assertEquals(output[1], 'var_2')
        output = USER.set_name_x(['my_var1', 'my_var2'], self.X, True)
        self.assertEquals(output[0], 'CONSTANT')
        self.assertEquals(output[1], 'my_var1')
        self.assertEquals(output[2], 'my_var2')
        output = USER.set_name_x(['my_var1', 'my_var2'], self.X, False)
        self.assertEquals(output[0], 'my_var1')
        self.assertEquals(output[1], 'my_var2')
    def test_set_name_yend(self):
        output = USER.set_name_yend([], self.X)
        self.assertEquals(output[0], 'endogenous_1')
        self.assertEquals(output[1], 'endogenous_2')
        output = USER.set_name_yend(None, self.X)
        self.assertEquals(output[0], 'endogenous_1')
        self.assertEquals(output[1], 'endogenous_2')
        end = self.X[:,0]
        end.shape = (self.X.shape[0], 1)
        output = USER.set_name_yend(None, end)
        self.assertEquals(output[0], 'endogenous_1')
        output = USER.set_name_yend(['my_var1', 'my_var2'], self.X)
        self.assertEquals(output[0], 'my_var1')
        self.assertEquals(output[1], 'my_var2')
    def test_set_name_q(self):
        output = USER.set_name_q([], self.X)
        self.assertEquals(output[0], 'instrument_1')
        self.assertEquals(output[1], 'instrument_2')
        output = USER.set_name_q(None, self.X)
        self.assertEquals(output[0], 'instrument_1')
        self.assertEquals(output[1], 'instrument_2')
        output = USER.set_name_q(['my_var1', 'my_var2'], self.X)
        self.assertEquals(output[0], 'my_var1')
        self.assertEquals(output[1], 'my_var2')
    def test_set_name_yend_sp(self):
        output = USER.set_name_yend_sp('my_variable')
        self.assertEquals(output, 'lag_my_variable')
    def test_set_name_q_sp(self):
        output = USER.set_name_q_sp(['my_var1', 'my_var2'], 1)
        self.assertEquals(output[0], 'lag1_my_var1')
        self.assertEquals(output[1], 'lag1_my_var2')
        output = USER.set_name_q_sp(['my_var1', 'my_var2'], 2)
        self.assertEquals(output[0], 'lag1_my_var1')
        self.assertEquals(output[1], 'lag1_my_var2')
        self.assertEquals(output[2], 'lag2_my_var1')
        self.assertEquals(output[3], 'lag2_my_var2')
    def test_set_name_h(self):
        output = USER.set_name_h(['my_var1', 'my_var2'], ['my_var3', 'my_var4'])
        self.assertEquals(output[0], 'my_var1')
        self.assertEquals(output[1], 'my_var2')
        self.assertEquals(output[2], 'my_var3')
        self.assertEquals(output[3], 'my_var4')

    def test_summary_results(self):
        # don't know how to test multiline string in python 2.6
        pass

class Test_Checkers(unittest.TestCase):
    def setUp(self):
        db=pysal.open("../../examples/columbus.dbf","r")
        y = np.array(db.by_col("CRIME"))
        self.y = y
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("HOVAL"))
        self.X = X
    
    def test_check_arrays(self):
        # X not an array
        self.assertRaises(Exception, USER.check_arrays, self.y, self.X)
        self.X = np.array(self.X)  # X and y wrong shape
        self.assertRaises(Exception, USER.check_arrays, self.y, self.X)
        self.X = self.X.T # y still wrong shape
        self.assertRaises(Exception, USER.check_arrays, self.y, self.X)
        self.y = np.reshape(self.y, (49,1))
        self.y = self.y[1:,:]  # X and y with different lengths
        self.assertRaises(Exception, USER.check_arrays, self.y, self.X)

    def test_check_weights(self):
        w = 4
        self.assertRaises(Exception, USER.check_weights, w, self.y)

    

suite = unittest.TestLoader().loadTestsFromTestCase(Test_OLS)
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(Test_Checkers))

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
