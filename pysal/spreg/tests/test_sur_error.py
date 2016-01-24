import unittest
import numpy as np
import pysal
from pysal.spreg.sur_utils import sur_dictxy,sur_dictZ
from pysal.spreg.sur_error import SURerrorML
from pysal.common import RTOL

PEGP = pysal.examples.get_path

def test_dic(actual, desired, rtol):
    for i in actual.keys():
        np.testing.assert_allclose(actual[i],desired[i],rtol)


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

        test_dic(reg.bSUR0,{0: np.array([[ 5.18423225],[ 0.67757925],
        [ 0.25706498]]), 1: np.array([[ 3.79731807],[ 1.02411196],[ 0.35895674]])},RTOL)
        test_dic(reg.bSUR,{0: np.array([[ 4.0222855 ],[ 0.88489646],[ 0.42402853]]),\
         1: np.array([[ 3.04923009],[ 1.10972634],[ 0.47075682]])},RTOL)
        test_dic(reg.sur_inf,{0: np.array([[  3.66921814e-01,   1.09622414e+01,   5.80436964e-28],\
        [  1.41290774e-01,   6.26294579e+00,   3.77771972e-10],\
        [  4.26795440e-02,   9.93517021e+00,   2.92675677e-23]]),\
         1: np.array([[  3.31399691e-01,   9.20106497e+00,   3.54419478e-20],\
        [  1.33525912e-01,   8.31094371e+00,   9.49439563e-17],\
        [  4.00409716e-02,   1.17568780e+01,   6.50970965e-32]])},RTOL)
        np.testing.assert_allclose(reg.lamols,np.array([[ 0.60205035],[ 0.56056348]]),RTOL)
        np.testing.assert_allclose(reg.lamsur,np.array([[ 0.54361986],[ 0.50445451]]),RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.31763719],\
       [ 0.31763719,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(5.1073696860799931, 1, 0.023824413482255974),\
         (1.9524745281321374, 1, 0.16232044613203933),\
         (0.79663667463065702, 1, 0.37210085476281407)],RTOL)
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

        test_dic(reg.bSUR,{0: np.array([[ 4.0222855 ],[ 0.88489646],[ 0.42402853]]),\
         1: np.array([[ 3.04923009],[ 1.10972634],[ 0.47075682]])},RTOL)
        test_dic(reg.sur_inf,{0: np.array([[  3.66921814e-01,   1.09622414e+01,   5.80436964e-28],\
        [  1.41290774e-01,   6.26294579e+00,   3.77771972e-10],\
        [  4.26795440e-02,   9.93517021e+00,   2.92675677e-23]]),\
         1: np.array([[  3.31399691e-01,   9.20106497e+00,   3.54419478e-20],\
        [  1.33525912e-01,   8.31094371e+00,   9.49439563e-17],\
        [  4.00409716e-02,   1.17568780e+01,   6.50970965e-32]])},RTOL)
        np.testing.assert_allclose(reg.vm, np.array([[  4.14625293e-04,   2.38494923e-05,  -3.48748935e-03,\
         -5.55994101e-04,  -1.63239040e-04],[  2.38494923e-05,   4.53642714e-04,  -2.00602452e-04,\
         -5.46893937e-04,  -3.10498019e-03],[ -3.48748935e-03,  -2.00602452e-04,   7.09989591e-01,\
          2.11105214e-01,   6.39785285e-02],[ -5.55994101e-04,  -5.46893937e-04,   2.11105214e-01,\
          3.42890248e-01,   1.91931389e-01],[ -1.63239040e-04,  -3.10498019e-03,   6.39785285e-02,\
          1.91931389e-01,   5.86933821e-01]]),RTOL)
        np.testing.assert_allclose(reg.lamsetp,(np.array([[ 0.02036235],\
        [ 0.02129889]]), np.array([[ 26.69730489],[ 23.68454458]]), np.array([[  5.05846638e-157],\
        [  5.20352683e-124]])),RTOL)
        np.testing.assert_allclose(reg.joinlam,(1207.81269, 2, 5.33096785e-263))
        np.testing.assert_allclose(reg.surchow,[(5.1073696860799931, 1, 0.023824413482255974),
        (1.9524745281321374, 1, 0.16232044613203933),
        (0.79663667463065702, 1, 0.37210085476281407)],RTOL)
        np.testing.assert_allclose(reg.likrlambda,(1014.0319285186415, 2, 6.3938800607190098e-221))
        np.testing.assert_allclose(reg.lrtest, (287.95821154104488, 1, 1.3849971230596533e-64))
        np.testing.assert_allclose(reg.lamtest,(1.86934297,  1,  0.17155035))

    def test_error_3eq(self): #Three equation example, unequal K
        y_var1 = ['HR60','HR70','HR80']
        x_var1 = [['RD60','PS60'],['RD70','PS70','UE70'],['RD80','PS80']]
        bigy1,bigX1,bigyvars1,bigXvars1 = sur_dictxy(self.db,y_var1,x_var1)
        reg = SURerrorML(bigy1,bigX1,self.w,name_bigy=bigyvars1,name_bigX=bigXvars1,\
            name_w="natqueen",name_ds="natregimes")        

        test_dic(reg.bSUR0,{0: np.array([[ 4.50407527],[ 2.39199682],[ 0.52723694]]), 1: np.array([[ 7.44509818],\
        [ 3.74968571],[ 1.28811685],[-0.23526451]]), 2: np.array([[ 6.92761614],[ 3.65423052],\
        [ 1.38247611]])},RTOL)
        test_dic(reg.bSUR,{0: np.array([[ 4.474891  ],[ 2.19004379],[ 0.59110509]]), 1: np.array([[ 7.15676612],\
       [ 3.49581077],[ 1.12846288],[-0.17133968]]), 2: np.array([[ 6.91550936],[ 3.69351192],\
       [ 1.40395543]])},RTOL)
        test_dic(reg.sur_inf,{0: np.array([[  1.34555859e-001,   3.32567528e+001,   1.63042050e-242],\
       [  1.20531820e-001,   1.81698362e+001,   8.94650541e-074],\
       [  1.09265711e-001,   5.40979573e+000,   6.30966781e-008]]),\
         1: np.array([[  2.95769241e-001,   2.41971271e+001,   2.38517181e-129],
       [  1.48214402e-001,   2.35861732e+001,   5.34381995e-123],
       [  1.34468661e-001,   8.39201319e+000,   4.77887269e-017],
       [  5.37833541e-002,  -3.18573719e+000,   1.44385693e-003]]),
         2: np.array([[  1.50052833e-001,   4.60871628e+001,   0.00000000e+000],\
       [  1.23633986e-001,   2.98745684e+001,   4.21160532e-196],\
       [  1.19498912e-001,   1.17486881e+001,   7.17236659e-032]])},RTOL)
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

