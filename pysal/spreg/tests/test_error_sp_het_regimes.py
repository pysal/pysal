import unittest
import pysal
import numpy as np
from pysal.spreg import error_sp_het_regimes as SP
from pysal.spreg.error_sp_het import GM_Error_Het, GM_Endog_Error_Het, GM_Combo_Het
from pysal.common import RTOL

class TestGM_Error_Het_Regimes(unittest.TestCase):
    def setUp(self):
        #Columbus:
        db=pysal.open(pysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("HOVAL"))
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        X2 = []
        X2.append(db.by_col("INC"))
        self.X2 = np.array(X2).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'
        self.r_var = 'NSA'
        self.regimes = db.by_col(self.r_var)
        #Artficial:
        n = 256
        self.n2 = n//2
        self.x_a1 = np.random.uniform(-10,10,(n,1))
        self.x_a2 = np.random.uniform(1,5,(n,1))
        self.q_a = self.x_a2 + np.random.normal(0,1,(n,1))
        self.x_a = np.hstack((self.x_a1,self.x_a2))
        self.y_a = np.dot(np.hstack((np.ones((n,1)),self.x_a)),np.array([[1],[0.5],[2]])) + np.random.normal(0,1,(n,1))
        latt = int(np.sqrt(n))
        self.w_a = pysal.lat2W(latt,latt)
        self.w_a.transform='r'
        self.regi_a = [0]*(n//2) + [1]*(n//2)
        self.w_a1 = pysal.lat2W(latt//2,latt)
        self.w_a1.transform='r'
        
    def test_model(self):
        reg = SP.GM_Error_Het_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array([[ 62.95986466],
       [ -0.15660795],
       [ -1.49054832],
       [ 60.98577615],
       [ -0.3358993 ],
       [ -0.82129289],
       [  0.54662719]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([-2.19031456])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 17.91629456])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 6
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.      ,   0.      ,   0.      ,   1.      ,  80.467003,  19.531   ]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        e = np.array([ 2.77847355])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my, RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy, RTOL)
        vm = np.array([  3.86154100e+01,  -2.51553730e-01,  -8.20138673e-01,
         1.71714184e+00,  -1.94929113e-02,   1.23118051e-01,
         0.00000000e+00])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        pr2 = 0.5515791216043385
        np.testing.assert_allclose(reg.pr2,pr2, RTOL)
        std_err = np.array([ 6.21412987,  0.15340022,  0.44060473,  7.6032169 ,  0.19353719,
        0.73621596,  0.13968272])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.04190799,  0.83779526],
       [ 0.5736724 ,  0.44880328],
       [ 0.62498575,  0.42920056]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.72341901308525713
        np.testing.assert_allclose(reg.chow.joint[0],chow_j, RTOL)

    def test_model_regi_error(self):
        #Columbus:
        reg = SP.GM_Error_Het_Regimes(self.y, self.X, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[ 60.74090229],
       [ -0.17492294],
       [ -1.33383387],
       [  0.68303064],
       [ 66.30374279],
       [ -0.31841139],
       [ -1.27502813],
       [  0.11515312]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 44.9411672 ,  -0.34343354,  -0.39946055,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([-0.05357818])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 15.77955818])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 0.70542044])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  3.11061225e-01,   5.77029704e-01],
       [  3.39747489e-01,   5.59975012e-01],
       [  3.86371771e-03,   9.50436364e-01],
       [  4.02884201e+00,   4.47286322e-02]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 4.7467070503995412
        np.testing.assert_allclose(reg.chow.joint[0],chow_j, RTOL)
        #Artficial:
        model = SP.GM_Error_Het_Regimes(self.y_a, self.x_a, self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = GM_Error_Het(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Error_Het(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas, RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm, RTOL)

    def test_model_endog(self):
        reg = SP.GM_Endog_Error_Het_Regimes(self.y, self.X2, self.yd, self.q, self.regimes, self.w)
        betas = np.array([[ 77.26679984],
       [  4.45992905],
       [ 78.59534391],
       [  0.41432319],
       [ -3.20196286],
       [ -1.13672283],
       [  0.2174965 ]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 20.50716917])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 25.13517175])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([-4.78118917])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n, RTOL)
        k = 6
        np.testing.assert_allclose(reg.k,k, RTOL)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        yend = np.array([[  0.      ,  80.467003]])
        np.testing.assert_allclose(reg.yend[0].toarray(),yend,RTOL)
        z = np.array([[  0.      ,   0.      ,   1.      ,  19.531   ,   0.      ,
         80.467003]])
        np.testing.assert_allclose(reg.z[0].toarray(),z,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my, RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy, RTOL)
        vm = np.array([ 509.66122149,  150.5845341 ,    9.64413821,    5.54782831,
        -80.95846045,   -2.25308524,   -3.2045214 ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        pr2 = 0.19776512679331681
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 22.57567765,  11.34616946,  17.43881791,   1.30953812,
         5.4830829 ,   0.74634612,   0.29973079])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.0022216 ,  0.96240654],
       [ 0.13127347,  0.7171153 ],
       [ 0.14367307,  0.70465645]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.2329971019087163
        np.testing.assert_allclose(reg.chow.joint[0],chow_j, RTOL)

    def test_model_endog_regi_error(self):
        #Columbus:
        reg = SP.GM_Endog_Error_Het_Regimes(self.y, self.X2, self.yd, self.q, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[  7.92747424e+01],
       [  5.78086230e+00],
       [ -3.83173581e+00],
       [  2.23210962e-01],
       [  8.26255251e+01],
       [  5.48294187e-01],
       [ -1.28432891e+00],
       [  3.57661629e-02]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([  7.55988579e+02,   2.53659722e+02,  -1.34288316e+02,
        -2.66141766e-01,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 25.73781918])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([-10.01183918])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 26.5449135])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[ 0.00998573,  0.92040097],
       [ 0.12660165,  0.72198192],
       [ 0.12737281,  0.72117171],
       [ 0.43507956,  0.50950696]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.3756768204399892
        np.testing.assert_allclose(reg.chow.joint[0],chow_j, RTOL)
        #Artficial:
        model = SP.GM_Endog_Error_Het_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = GM_Endog_Error_Het(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Endog_Error_Het(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas, RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm, RTOL)

    def test_model_combo(self):
        reg = SP.GM_Combo_Het_Regimes(self.y, self.X2, self.regimes, self.yd, self.q, w=self.w)
        betas = np.array([[  3.69372678e+01],
       [ -8.29474998e-01],
       [  3.08667517e+01],
       [ -7.23753444e-01],
       [ -3.01900940e-01],
       [ -2.21328949e-01],
       [  6.41902155e-01],
       [ -2.45714919e-02]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 0.94039246])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 0.8737864])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy_e = np.array([ 18.68732105])
        np.testing.assert_allclose(reg.predy_e[0],predy_e,RTOL)
        predy = np.array([ 14.78558754])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n, RTOL)
        k = 7
        np.testing.assert_allclose(reg.k,k, RTOL)
        y = np.array([ 15.72598])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([[  0.   ,   0.   ,   1.   ,  19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(),x,RTOL)
        yend = np.array([[  0.       ,  80.467003 ,  24.7142675]])
        np.testing.assert_allclose(reg.yend[0].toarray(),yend,RTOL)
        z = np.array([[  0.       ,   0.       ,   1.       ,  19.531    ,   0.       ,
         80.467003 ,  24.7142675]])
        np.testing.assert_allclose(reg.z[0].toarray(),z,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my, RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy, RTOL)
        vm = np.array([ 71.26851365,  -0.58278032,  50.53169815,  -0.74561147,
        -0.79510274,  -0.10823496,  -0.98141395,   1.16575965])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        pr2 = 0.6504148883602958
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.527136896994038
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 8.44206809,  0.72363219,  9.85790968,  0.77218082,  0.34084146,
        0.21752916,  0.14371614,  0.39226478])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.54688708,  0.45959243],
       [ 0.01035136,  0.91896175],
       [ 0.03981108,  0.84185042]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.78070369988354349
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_combo_regi_error(self):
        #Columbus:
        reg = SP.GM_Combo_Het_Regimes(self.y, self.X2, self.regimes, self.yd, self.q, w=self.w, regime_lag_sep=True, regime_err_sep=True)
        betas = np.array([[ 42.01151458],
       [ -0.13917151],
       [ -0.65300184],
       [  0.54737064],
       [  0.2629229 ],
       [ 34.21569751],
       [ -0.15236089],
       [ -0.49175217],
       [  0.65733173],
       [ -0.07713581]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 77.49519689,   0.57226879,  -1.18856422,  -1.28088712,
         0.866752  ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 7.81039418])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 7.91558582])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 7.22996911])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  1.90869079e-01,   6.62194273e-01],
       [  4.56118982e-05,   9.94611401e-01],
       [  3.12104263e-02,   8.59771748e-01],
       [  1.56368204e-01,   6.92522476e-01],
       [  7.52928732e-01,   3.85550558e-01]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.1316136604755913
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
        #Artficial:
        model = SP.GM_Combo_Het_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True, regime_lag_sep=True)
        model1 = GM_Combo_Het(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Combo_Het(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        #was this supposed to get tested?

if __name__ == '__main__':
    unittest.main()
