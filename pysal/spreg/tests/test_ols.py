import unittest
import numpy as np
import pysal
from pysal.spreg import utils
import pysal.spreg as EC

PEGP = pysal.examples.get_path

class TestBaseOLS(unittest.TestCase):
    def setUp(self):
        db = pysal.open(PEGP('columbus.dbf'),'r')
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = pysal.weights.rook_from_shapefile(PEGP("columbus.shp"))

    def test_ols(self):
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        ols = EC.ols.BaseOLS(self.y,self.X)
        np.testing.assert_array_almost_equal(ols.betas, np.array([[
            46.42818268], [  0.62898397], [ -0.48488854]]))
        vm = np.array([[  1.74022453e+02,  -6.52060364e+00,  -2.15109867e+00],
           [ -6.52060364e+00,   2.87200008e-01,   6.80956787e-02],
           [ -2.15109867e+00,   6.80956787e-02,   3.33693910e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm,6)

    def test_ols_white1(self):
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        ols = EC.ols.BaseOLS(self.y,self.X,robust='white', sig2n_k=True)
        np.testing.assert_array_almost_equal(ols.betas, np.array([[
            46.42818268], [  0.62898397], [ -0.48488854]]))
        vm = np.array([[  2.05819450e+02,  -6.83139266e+00,  -2.64825846e+00],
       [ -6.83139266e+00,   2.58480813e-01,   8.07733167e-02],
       [ -2.64825846e+00,   8.07733167e-02,   3.75817181e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm,6)

    def test_ols_white2(self):
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        ols = EC.ols.BaseOLS(self.y,self.X,robust='white', sig2n_k=False)
        np.testing.assert_array_almost_equal(ols.betas, np.array([[
            46.42818268], [  0.62898397], [ -0.48488854]]))
        vm = np.array([[  1.93218259e+02,  -6.41314413e+00,  -2.48612018e+00],
       [ -6.41314413e+00,   2.42655457e-01,   7.58280116e-02],
       [ -2.48612018e+00,   7.58280116e-02,   3.52807966e-02]])
        np.testing.assert_array_almost_equal(ols.vm, vm,6)

    def test_OLS(self):
        ols = EC.OLS(self.y, self.X, self.w, spat_diag=True, moran=True, \
                white_test=True, name_y='home value', name_x=['income','crime'], \
                name_ds='columbus')
        
        np.testing.assert_array_almost_equal(ols.aic, \
                408.73548964604873 ,7)
        np.testing.assert_array_almost_equal(ols.ar2, \
                0.32123239427957662 ,7)
        np.testing.assert_array_almost_equal(ols.betas, \
                np.array([[ 46.42818268], [  0.62898397], \
                    [ -0.48488854]]), 7) 
        bp = np.array([2, 5.7667905131212587, 0.05594449410070558])
        ols_bp = np.array([ols.breusch_pagan['df'], ols.breusch_pagan['bp'], ols.breusch_pagan['pvalue']])
        np.testing.assert_array_almost_equal(bp, ols_bp, 7)
        np.testing.assert_array_almost_equal(ols.f_stat, \
            (12.358198885356581, 5.0636903313953024e-05), 7)
        jb = np.array([2, 39.706155069114878, 2.387360356860208e-09])
        ols_jb = np.array([ols.jarque_bera['df'], ols.jarque_bera['jb'], ols.jarque_bera['pvalue']])
        np.testing.assert_array_almost_equal(ols_jb,jb, 7)
        white = np.array([5, 2.90606708, 0.71446484])
        ols_white = np.array([ols.white['df'], ols.white['wh'], ols.white['pvalue']])
        np.testing.assert_array_almost_equal(ols_white,white, 7)
        np.testing.assert_equal(ols.k,  3)
        kb = {'df': 2, 'kb': 2.2700383871478675, 'pvalue': 0.32141595215434604}
        for key in kb:
            self.assertAlmostEqual(ols.koenker_bassett[key],  kb[key], 7)
        np.testing.assert_array_almost_equal(ols.lm_error, \
            (4.1508117035117893, 0.041614570655392716),7)
        np.testing.assert_array_almost_equal(ols.lm_lag, \
            (0.98279980617162233, 0.32150855529063727), 7)
        np.testing.assert_array_almost_equal(ols.lm_sarma, \
                (4.3222725729143736, 0.11519415308749938), 7)
        np.testing.assert_array_almost_equal(ols.logll, \
                -201.3677448230244 ,7)
        np.testing.assert_array_almost_equal(ols.mean_y, \
            38.436224469387746,7)
        np.testing.assert_array_almost_equal(ols.moran_res[0], \
            0.20373540938,7)
        np.testing.assert_array_almost_equal(ols.moran_res[1], \
            2.59180452208,7)
        np.testing.assert_array_almost_equal(ols.moran_res[2], \
            0.00954740031251,7)
        np.testing.assert_array_almost_equal(ols.mulColli, \
            12.537554873824675 ,7)
        np.testing.assert_equal(ols.n,  49)
        np.testing.assert_equal(ols.name_ds,  'columbus')
        np.testing.assert_equal(ols.name_gwk,  None)
        np.testing.assert_equal(ols.name_w,  'unknown')
        np.testing.assert_equal(ols.name_x,  ['CONSTANT', 'income', 'crime'])
        np.testing.assert_equal(ols.name_y,  'home value')
        np.testing.assert_array_almost_equal(ols.predy[3], np.array([
            33.53969014]),7)
        np.testing.assert_array_almost_equal(ols.r2, \
                0.34951437785126105 ,7)
        np.testing.assert_array_almost_equal(ols.rlm_error, \
                (3.3394727667427513, 0.067636278225568919),7)
        np.testing.assert_array_almost_equal(ols.rlm_lag, \
            (0.17146086940258459, 0.67881673703455414), 7)
        np.testing.assert_equal(ols.robust,  'unadjusted')
        np.testing.assert_array_almost_equal(ols.schwarz, \
            414.41095054038061,7 )
        np.testing.assert_array_almost_equal(ols.sig2, \
            231.4568494392652,7 )
        np.testing.assert_array_almost_equal(ols.sig2ML, \
            217.28602192257551,7 )
        np.testing.assert_array_almost_equal(ols.sig2n, \
                217.28602192257551, 7)
 
        np.testing.assert_array_almost_equal(ols.t_stat[2][0], \
                -2.65440864272,7)
        np.testing.assert_array_almost_equal(ols.t_stat[2][1], \
                0.0108745049098,7)

if __name__ == '__main__':
    unittest.main()
