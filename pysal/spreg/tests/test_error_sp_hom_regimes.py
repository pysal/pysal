import unittest
import pysal
import numpy as np
from pysal.spreg import error_sp_hom_regimes as SP
from pysal.spreg.error_sp_hom import GM_Error_Hom, GM_Endog_Error_Hom, GM_Combo_Hom
from pysal.common import RTOL

class TestGM_Error_Hom_Regimes(unittest.TestCase):
    def setUp(self):
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
        self.regi_a = [0]*(n//2) + [1]*(n//2) ##must be floos!
        self.w_a1 = pysal.lat2W(latt//2,latt)
        self.w_a1.transform='r'
        
    def test_model(self):
        reg = SP.GM_Error_Hom_Regimes(self.y, self.X, self.regimes, self.w, A1='het')
        betas = np.array([[ 62.95986466],
       [ -0.15660795],
       [ -1.49054832],
       [ 60.98577615],
       [ -0.3358993 ],
       [ -0.82129289],
       [  0.54033921]])
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
        e = np.array([ 2.72131648])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([ 49.16245801,  -0.12493165,  -1.89294614,   5.71968257,
        -0.0571525 ,   0.05745855,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        sig2 = 96.96108341267626
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.5515791216023577
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        std_err = np.array([ 7.01159454,  0.20701411,  0.56905515,  7.90537942,  0.10268949,
        0.56660879,  0.15659504])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.03888544,  0.84367579],
       [ 0.61613446,  0.43248738],
       [ 0.72632441,  0.39407719]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.92133276766189676
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_regi_error(self):
        #Artficial:
        model = SP.GM_Error_Hom_Regimes(self.y_a, self.x_a, self.regi_a, w=self.w_a, regime_err_sep=True, A1='het')
        model1 = GM_Error_Hom(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a[0:(self.n2)], w=self.w_a1, A1='het')
        model2 = GM_Error_Hom(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a[(self.n2):], w=self.w_a1, A1='het')
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm, RTOL)
        #Columbus:
        reg = SP.GM_Error_Hom_Regimes(self.y, self.X2, self.regimes, self.w, regime_err_sep=True, A1='het')
        betas = np.array([[ 60.66668194],
       [ -1.72708492],
       [  0.62170311],
       [ 61.4526885 ],
       [ -1.90700858],
       [  0.1102755 ]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 45.57956967,  -1.65365774,   0.        ,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([-8.48092392])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 24.20690392])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([-8.33982604])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[ 0.0050892 ,  0.94312823],
       [ 0.05746619,  0.81054651],
       [ 1.65677138,  0.19803981]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.7914221673031792
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_endog(self):
        reg = SP.GM_Endog_Error_Hom_Regimes(self.y, self.X2, self.yd, self.q, self.regimes, self.w, A1='het')
        betas = np.array([[ 77.26679984],
       [  4.45992905],
       [ 78.59534391],
       [  0.41432319],
       [ -3.20196286],
       [ -1.13672283],
       [  0.22178164]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 20.50716917])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 25.22635318])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([-4.78118917])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 6
        np.testing.assert_allclose(reg.k,k,RTOL)
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
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([ 403.76852704,   69.06920553,   19.8388512 ,    3.62501395,
        -40.30472224,   -1.6601927 ,   -1.64319352])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        pr2 = 0.19776512679498906
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        sig2 = 644.23810259214
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        std_err = np.array([ 20.09399231,   7.03617703,  23.64968032,   2.176846  ,
         3.40352278,   0.92377997,   0.24462006])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.00191145,  0.96512749],
       [ 0.31031517,  0.57748685],
       [ 0.34994619,  0.55414359]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.248410480025556
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_endog_regi_error(self):
        #Columbus:
        reg = SP.GM_Endog_Error_Hom_Regimes(self.y, self.X2, self.yd, self.q, self.regimes, self.w, regime_err_sep=True, A1='het')
        betas = np.array([[  7.92747424e+01],
       [  5.78086230e+00],
       [ -3.83173581e+00],
       [  2.14725610e-01],
       [  8.26255251e+01],
       [  5.48294187e-01],
       [ -1.28432891e+00],
       [  2.98658172e-02]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 867.50930457,  161.04430783,  -92.35637083,   -1.13838767,
          0.        ,    0.        ,    0.        ,    0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 25.73781918])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([-10.01183918])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([26.41176711])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[ 0.00909777,  0.92401124],
       [ 0.24034941,  0.62395386],
       [ 0.24322564,  0.62188603],
       [ 0.32572159,  0.5681893 ]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 1.4485058522307526
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
        #Artficial:
        model = SP.GM_Endog_Error_Hom_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True, A1='het')
        model1 = GM_Endog_Error_Hom(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1, A1='het')
        model2 = GM_Endog_Error_Hom(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1, A1='het')
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm, RTOL)

    def test_model_combo(self):
        reg = SP.GM_Combo_Hom_Regimes(self.y, self.X2, self.regimes, self.yd, self.q, w=self.w, A1='het')
        betas = np.array([[ 36.93726782],
       [ -0.829475  ],
       [ 30.86675168],
       [ -0.72375344],
       [ -0.30190094],
       [ -0.22132895],
       [  0.64190215],
       [ -0.07314671]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 0.94039246])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 0.74211331])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy_e = np.array([ 18.68732105])
        np.testing.assert_allclose(reg.predy_e[0],predy_e,RTOL)
        predy = np.array([ 14.78558754])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 7
        np.testing.assert_allclose(reg.k,k,RTOL)
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
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([ 111.54419614,   -0.23476709,   83.37295278,   -1.74452409,
         -1.60256796,   -0.13151396,   -1.43857915,    2.19420848])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        sig2 = 95.57694234438294
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.6504148883591536
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.5271368969923579
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 10.56144858,   0.93986958,  11.52977369,   0.61000358,
         0.44419535,   0.16191882,   0.1630835 ,   0.41107528])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        chow_r = np.array([[ 0.47406771,  0.49112176],
       [ 0.00879838,  0.92526827],
       [ 0.02943577,  0.86377672]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.59098559257602923
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)

    def test_model_combo_regi_error(self):
        #Columbus:
        reg = SP.GM_Combo_Hom_Regimes(self.y, self.X2, self.regimes, self.yd, self.q, w=self.w, regime_lag_sep=True, regime_err_sep=True, A1='het')
        betas = np.array([[  4.20115146e+01],
       [ -1.39171512e-01],
       [ -6.53001838e-01],
       [  5.47370644e-01],
       [  2.61860326e-01],
       [  3.42156975e+01],
       [ -1.52360889e-01],
       [ -4.91752171e-01],
       [  6.57331733e-01],
       [ -2.68716241e-02]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 154.23356187,    2.99104716,   -3.29036767,   -2.473113  ,
          1.65247551,    0.        ,    0.        ,    0.        ,
          0.        ,    0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 7.81039418])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 7.91558582])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 7.60819283])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  9.59590706e-02,   7.56733881e-01],
       [  6.53130455e-05,   9.93551847e-01],
       [  4.65270134e-02,   8.29220655e-01],
       [  7.68939379e-02,   7.81551631e-01],
       [  5.04560098e-01,   4.77503278e-01]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.74134991257940286
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
        #Artficial:
        model = SP.GM_Combo_Hom_Regimes(self.y_a, self.x_a1, yend=self.x_a2, q=self.q_a, regimes=self.regi_a, w=self.w_a, regime_err_sep=True, regime_lag_sep=True, A1='het')
        model1 = GM_Combo_Hom(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a1[0:(self.n2)], yend=self.x_a2[0:(self.n2)], q=self.q_a[0:(self.n2)], w=self.w_a1, A1='het')
        model2 = GM_Combo_Hom(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a1[(self.n2):], yend=self.x_a2[(self.n2):], q=self.q_a[(self.n2):], w=self.w_a1, A1='het')
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))


if __name__ == '__main__':
    unittest.main()
