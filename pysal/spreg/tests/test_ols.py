import unittest
import numpy as np
import pysal

class TestBaseOLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T

    def test_ols(self):
        ols = pysal.spreg.ols.BaseOLS(self.y,self.X)
        np.testing.assert_array_almost_equal(ols.betas, np.array([[
            46.42818268], [  0.62898397], [ -0.48488854]]))
        vm = np.array([[  1.74022453e+02,  -6.52060364e+00,  -2.15109867e+00],
           [ -6.52060364e+00,   2.87200008e-01,   6.80956787e-02],
           [ -2.15109867e+00,   6.80956787e-02,   3.33693910e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm,6)

    def test_OLS(self):
        ols = pysal.spreg.ols.OLS(self.y,self.X)
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -2.65440864272,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.0108745049098,7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.34951437785126105 ,7)



if __name__ == '__main__':
    unittest.main()
