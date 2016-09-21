import unittest
import numpy as np
import pysal
from pysal.spreg.sur_utils import sur_dictxy,sur_dictZ
from pysal.spreg.sur_error import SURerrorML
from pysal.common import RTOL
ATOL = 1e-12

PEGP = pysal.examples.get_path

def dict_compare(actual, desired, rtol=RTOL, atol=ATOL):
    for i in actual.keys():
        np.testing.assert_allclose(actual[i],desired[i],rtol=rtol, atol=atol)


class Test_SUR_error(unittest.TestCase):
    def setUp(self):
        self.db = pysal.open(pysal.examples.get_path('NAT.dbf'),'r')
        self.w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
        self.w.transform = 'r'


    def test_error(self): #2 equations
        y_var0 = ['HR80','HR90']
        x_var0 = [['PS80','UE80'],['PS90','UE90']]
        bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(self.db,y_var0,x_var0)
        reg = SURerrorML(bigy0,bigX0,self.w,\
                  name_bigy=bigyvars0,name_bigX=bigXvars0,spat_diag=True,\
                  name_w="natqueen",name_ds="natregimes",nonspat_diag=False)

        dict_compare(reg.bSUR0,{0: np.array([[ 5.18423225],[ 0.67757925],
        [ 0.25706498]]), 1: np.array([[ 3.79731807],[ 1.02411196],[ 0.35895674]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 4.0222855 ],[ 0.88489646],[ 0.42402853]]),\
         1: np.array([[ 3.04923009],[ 1.10972634],[ 0.47075682]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  3.669218e-01,   1.096224e+01,   5.804195e-28],\
       [  1.412908e-01,   6.262946e+00,   3.777726e-10],\
       [  4.267954e-02,   9.935169e+00,   2.926783e-23]]),\
         1: np.array([[  3.31399691e-01,   9.20106497e+00,   3.54419478e-20],\
        [  1.33525912e-01,   8.31094371e+00,   9.49439563e-17],\
        [  4.00409716e-02,   1.17568780e+01,   6.50970965e-32]])},rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.lamols,np.array([[ 0.60205035],[ 0.56056348]]),RTOL)
        np.testing.assert_allclose(reg.lamsur,np.array([[ 0.54361986],[ 0.50445451]]),RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.31763719],\
       [ 0.31763719,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(5.1073696860799931, 1, 0.023824413482255974),\
         (1.9524745281321374, 1, 0.16232044613203933),\
         (0.79663667463065702, 1, 0.37210085476281407)],rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.llik,-19860.067987395596)
        np.testing.assert_allclose(reg.errllik,-19497.031128906794)
        np.testing.assert_allclose(reg.surerrllik,-19353.052023136348)
        np.testing.assert_allclose(reg.likrlambda, (1014.0319285186415, 2, 6.3938800607190098e-221))

    def test_error_vm(self): #Asymptotic variance matrix
        y_var0 = ['HR80','HR90']
        x_var0 = [['PS80','UE80'],['PS90','UE90']]
        bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(self.db,y_var0,x_var0)
        reg = SURerrorML(bigy0,bigX0,self.w,spat_diag=True,vm=True,\
                  name_bigy=bigyvars0,name_bigX=bigXvars0,\
                  name_w="natqueen",name_ds="natregimes")

        dict_compare(reg.bSUR,{0: np.array([[ 4.0222855 ],[ 0.88489646],[ 0.42402853]]),\
         1: np.array([[ 3.04923009],[ 1.10972634],[ 0.47075682]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  3.669218e-01,   1.096224e+01,   5.804195e-28],\
       [  1.412908e-01,   6.262946e+00,   3.777726e-10],\
       [  4.267954e-02,   9.935169e+00,   2.926783e-23]]),\
         1: np.array([[  3.31399691e-01,   9.20106497e+00,   3.54419478e-20],\
        [  1.33525912e-01,   8.31094371e+00,   9.49439563e-17],\
        [  4.00409716e-02,   1.17568780e+01,   6.50970965e-32]])},rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.vm, np.array([[  4.14625293e-04,   2.38494923e-05,  -3.48748935e-03,\
         -5.55994101e-04,  -1.63239040e-04],[  2.38494923e-05,   4.53642714e-04,  -2.00602452e-04,\
         -5.46893937e-04,  -3.10498019e-03],[ -3.48748935e-03,  -2.00602452e-04,   7.09989591e-01,\
          2.11105214e-01,   6.39785285e-02],[ -5.55994101e-04,  -5.46893937e-04,   2.11105214e-01,\
          3.42890248e-01,   1.91931389e-01],[ -1.63239040e-04,  -3.10498019e-03,   6.39785285e-02,\
          1.91931389e-01,   5.86933821e-01]]),RTOL)
        np.testing.assert_allclose(reg.lamsetp,(np.array([[ 0.02036235],\
        [ 0.02129889]]), np.array([[ 26.69730489],[ 23.68454458]]), np.array([[  5.059048e-157],\
        [  5.202838e-124]])),rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.joinlam,(1207.81269, 2, 5.330924e-263), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.surchow,[(5.1073696860799931, 1, 0.023824413482255974),
        (1.9524745281321374, 1, 0.16232044613203933),
        (0.79663667463065702, 1, 0.37210085476281407)],rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.likrlambda, (1014.0319285186415, 2, 6.3938800607190098e-221), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.lrtest, (287.95821154104488, 1, 1.3849971230596533e-64), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.lamtest, (1.8693306894921564, 1, 0.17155175615429052), rtol=RTOL, atol=ATOL)

    def test_error_3eq(self): #Three equation example, unequal K
        y_var1 = ['HR60','HR70','HR80']
        x_var1 = [['RD60','PS60'],['RD70','PS70','UE70'],['RD80','PS80']]
        bigy1,bigX1,bigyvars1,bigXvars1 = sur_dictxy(self.db,y_var1,x_var1)
        reg = SURerrorML(bigy1,bigX1,self.w,name_bigy=bigyvars1,name_bigX=bigXvars1,\
            name_w="natqueen",name_ds="natregimes")        

        dict_compare(reg.bSUR0,{0: np.array([[ 4.50407527],[ 2.39199682],[ 0.52723694]]), 1: np.array([[ 7.44509818],\
        [ 3.74968571],[ 1.28811685],[-0.23526451]]), 2: np.array([[ 6.92761614],[ 3.65423052],\
        [ 1.38247611]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 4.474891  ],[ 2.19004379],[ 0.59110509]]), 1: np.array([[ 7.15676612],\
       [ 3.49581077],[ 1.12846288],[-0.17133968]]), 2: np.array([[ 6.91550936],[ 3.69351192],\
       [ 1.40395543]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  1.345557e-001,   3.325679e+001,   1.628656e-242],\
       [  1.205317e-001,   1.816985e+001,   8.943986e-074],\
       [  1.092657e-001,   5.409796e+000,   6.309653e-008]]),\
         1: np.array([[  2.957692e-001,   2.419713e+001,   2.384951e-129],\
       [  1.482144e-001,   2.358618e+001,   5.343318e-123],\
       [  1.344687e-001,   8.392014e+000,   4.778835e-017],\
       [  5.378335e-002,  -3.185738e+000,   1.443854e-003]]),\
         2: np.array([[  1.500528e-001,   4.608718e+001,   0.000000e+000],\
       [  1.236340e-001,   2.987457e+001,   4.210941e-196],\
       [  1.194989e-001,   1.174869e+001,   7.172248e-032]])},rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(reg.lamols,np.array([[ 0.4248829 ],[ 0.46428101],[ 0.42823999]]),RTOL)
        np.testing.assert_allclose(reg.lamsur,np.array([[ 0.36137603],[ 0.38321666],[ 0.37183716]]),RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.,          0.24563253,  0.14986527],\
        [ 0.24563253,  1.,          0.25945021],[ 0.14986527,  0.25945021,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.llik,-28695.767676078722)
        np.testing.assert_allclose(reg.errllik,-28593.569427945633)
        np.testing.assert_allclose(reg.surerrllik,-28393.703607018397)
        np.testing.assert_allclose(reg.lrtest,(399.7316418544724, 3, 2.5309501580053097e-86))


if __name__ == '__main__':
    unittest.main()

