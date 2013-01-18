import unittest
import numpy as np
import pysal
from pysal.spreg import diagnostics
from pysal.spreg.ols import OLS 


# create regression object used by all the tests below
db = pysal.open(pysal.examples.get_path("columbus.dbf"), "r")
y = np.array(db.by_col("CRIME"))
y = np.reshape(y, (49,1))
X = []
X.append(db.by_col("INC"))
X.append(db.by_col("HOVAL"))
X = np.array(X).T
reg = OLS(y,X)


class TestFStat(unittest.TestCase):
    def test_f_stat(self):
        obs = diagnostics.f_stat(reg)
        exp = (28.385629224695, 0.000000009341)
        for i in range(2):
            self.assertAlmostEquals(obs[i],exp[i])

class TestTStat(unittest.TestCase):
    def test_t_stat(self):
        obs = diagnostics.t_stat(reg)
        exp = [(14.490373143689094, 9.2108899889173982e-19),
               (-4.7804961912965762, 1.8289595070843232e-05),
               (-2.6544086427176916, 0.010874504909754612)]
        for i in range(3):
            for j in range(2):
                self.assertAlmostEquals(obs[i][j],exp[i][j])

class TestR2(unittest.TestCase):
    def test_r2(self):
        obs = diagnostics.r2(reg)
        exp = 0.55240404083742334
        self.assertAlmostEquals(obs,exp)

class TestAr2(unittest.TestCase):
    def test_ar2(self):
        obs = diagnostics.ar2(reg)
        exp = 0.5329433469607896
        self.assertAlmostEquals(obs,exp)

class TestSeBetas(unittest.TestCase):
    def test_se_betas(self):
        obs = diagnostics.se_betas(reg)
        exp = np.array([4.73548613, 0.33413076, 0.10319868])
        np.testing.assert_array_almost_equal(obs,exp)

class TestLogLikelihood(unittest.TestCase):
    def test_log_likelihood(self):
        obs = diagnostics.log_likelihood(reg)
        exp = -187.3772388121491
        self.assertAlmostEquals(obs,exp)

class TestAkaike(unittest.TestCase):
    def test_akaike(self):
        obs = diagnostics.akaike(reg)
        exp = 380.7544776242982
        self.assertAlmostEquals(obs,exp)

class TestSchwarz(unittest.TestCase):
    def test_schwarz(self):
        obs = diagnostics.schwarz(reg)
        exp = 386.42993851863008
        self.assertAlmostEquals(obs,exp)

class TestConditionIndex(unittest.TestCase):
    def test_condition_index(self):
        obs = diagnostics.condition_index(reg)
        exp = 6.541827751444
        self.assertAlmostEquals(obs,exp)

class TestJarqueBera(unittest.TestCase):
    def test_jarque_bera(self):
        obs = diagnostics.jarque_bera(reg)
        exp = {'df':2, 'jb':1.835752520076, 'pvalue':0.399366291249}
        self.assertEquals(obs['df'],exp['df'])
        self.assertAlmostEquals(obs['jb'],exp['jb'])
        self.assertAlmostEquals(obs['pvalue'],exp['pvalue'])

class TestBreuschPagan(unittest.TestCase):
    def test_breusch_pagan(self):
        obs = diagnostics.breusch_pagan(reg)
        exp = {'df':2, 'bp':7.900441675960, 'pvalue':0.019250450075}
        self.assertEquals(obs['df'],exp['df'])
        self.assertAlmostEquals(obs['bp'],exp['bp'])
        self.assertAlmostEquals(obs['pvalue'],exp['pvalue'])

class TestWhite(unittest.TestCase):
    def test_white(self):
        obs = diagnostics.white(reg)
        exp = {'df':5, 'wh':19.946008239903, 'pvalue':0.001279222817}
        self.assertEquals(obs['df'],exp['df'])
        self.assertAlmostEquals(obs['wh'],exp['wh'])
        self.assertAlmostEquals(obs['pvalue'],exp['pvalue'])

class TestKoenkerBassett(unittest.TestCase):
    def test_koenker_bassett(self):
        obs = diagnostics.koenker_bassett(reg)
        exp = {'df':2, 'kb':5.694087931707, 'pvalue':0.058015563638}
        self.assertEquals(obs['df'],exp['df'])
        self.assertAlmostEquals(obs['kb'],exp['kb'])
        self.assertAlmostEquals(obs['pvalue'],exp['pvalue'])

class TestVif(unittest.TestCase):
    def test_vif(self):
        obs = diagnostics.vif(reg)
        exp = [(0.0, 0.0),  # note [0][1] should actually be infiniity...
               (1.3331174971891975, 0.75012142748740696),
               (1.3331174971891973, 0.75012142748740707)]
        for i in range(1,3):
            for j in range(2):
                self.assertAlmostEquals(obs[i][j],exp[i][j])

class TestConstantCheck(unittest.TestCase):
    def test_constant_check(self):
        obs = diagnostics.constant_check(reg.x)
        exp = True
        self.assertEquals(obs,exp)


if __name__ == '__main__':
    unittest.main()
