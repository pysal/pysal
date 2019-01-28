import unittest
import numpy as np
from ..sur_utils import sur_dictxy, sur_dictZ
from ..sur import SUR, ThreeSLS
import pysal.lib
from pysal.lib.common import RTOL

PEGP = pysal.lib.examples.get_path

def dict_compare(actual, desired, rtol, atol=1e-7):
    for i in actual.keys():
        np.testing.assert_allclose(actual[i],desired[i],rtol,atol=atol)


class Test_SUR(unittest.TestCase):
    def setUp(self):
        self.db = pysal.lib.io.open(pysal.lib.examples.get_path('NAT.dbf'),'r')
        self.w = pysal.lib.weights.Queen.from_shapefile(pysal.lib.examples.get_path("NAT.shp"))
        self.w.transform = 'r'


    def test_SUR(self): #2 equations, same K in each, two-step estimation
        y_var0 = ['HR80','HR90']
        x_var0 = [['PS80','UE80'],['PS90','UE90']]
        bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(self.db,y_var0,x_var0)
        reg = SUR(bigy0,bigX0,name_bigy=bigyvars0,name_bigX=bigXvars0)

        dict_compare(reg.bOLS,{0: np.array([[ 5.39719146],[ 0.6973813 ],\
        [ 0.22566378]]), 1: np.array([[ 1.80829725],[ 1.03504143],[ 0.6582483 ]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 5.13907179],[ 0.67764814],\
        [ 0.26372397]]), 1: np.array([[ 3.61394031],[ 1.02607147],[ 0.38654993]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  2.62467257e-01,   1.95798587e+01,   2.29656805e-85],\
        [  1.21957836e-01,   5.55641325e+00,   2.75374482e-08],\
        [  3.43183797e-02,   7.68462769e+00,   1.53442563e-14]]),\
         1: np.array([[  2.53499643e-01,   1.42561949e+01,   4.10220329e-46],\
        [  1.12166227e-01,   9.14777552e+00,   5.81179115e-20],\
        [  3.41995564e-02,   1.13027760e+01,   1.27134462e-29]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.46954842],\
        [ 0.46954842,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(26.72917022127309, 1, 2.3406126054510838e-07),\
        (8.2409218385398244, 1, 0.0040956326095295649),\
         (9.3837654127686712, 1, 0.002189154327032255)],RTOL)

    def test_SUR_iter(self): #2 equations, same K in each, iterated estimation, spatial test
        y_var0 = ['HR80','HR90']
        x_var0 = [['PS80','UE80'],['PS90','UE90']]
        bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(self.db,y_var0,x_var0)
        reg = SUR(bigy0,bigX0,w=self.w,nonspat_diag=True,spat_diag=True,iter=True,verbose=False,\
            name_bigy=bigyvars0,name_bigX=bigXvars0)

        dict_compare(reg.bOLS,{0: np.array([[ 5.39719146],[ 0.6973813 ],\
        [ 0.22566378]]), 1: np.array([[ 1.80829725],[ 1.03504143],[ 0.6582483 ]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 5.18423225],[ 0.67757925],\
        [ 0.25706498]]), 1: np.array([[ 3.79731807],[ 1.02411196],[ 0.35895674]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  2.59392406e-01,   1.99860602e+01,   7.28237551e-89],\
        [  1.21911330e-01,   5.55796781e+00,   2.72933704e-08],\
        [  3.38051365e-02,   7.60431727e+00,   2.86411830e-14]]),\
         1: np.array([[  2.53108919e-01,   1.50027035e+01,   7.04886598e-51],\
        [  1.13329850e-01,   9.03655976e+00,   1.61679985e-19],\
        [  3.40440433e-02,   1.05438928e+01,   5.42075621e-26]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.       ,  0.5079133],
        [ 0.5079133,  1.       ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(23.457203761752844, 1, 1.2772356421778157e-06),\
        (8.6998292497532859, 1, 0.0031823985960753743),\
        (6.8426866249389589, 1, 0.0089004881389025351)],RTOL)
        np.testing.assert_allclose(reg.llik,-19860.067987395596)
        np.testing.assert_allclose(reg.lmtest,(680.16759754291365, 1, 6.144389240997126e-150))
        np.testing.assert_allclose(reg.lrtest, (854.18095147295708, 1, 8.966465468792485e-188))
        np.testing.assert_allclose(reg.lmEtest,(1270.87724, 2, 1.07773778e-276))

    def test_SUR_3eq(self): #3 equations, different K, iterated estimation, spatial test
        y_var1 = ['HR60','HR70','HR80']
        x_var1 = [['RD60','PS60'],['RD70','PS70','UE70'],['RD80','PS80']]
        bigy1,bigX1,bigyvars1,bigXvars1 = sur_dictxy(self.db,y_var1,x_var1)
        reg = SUR(bigy1,bigX1,w=self.w,spat_diag=True,iter=True,verbose=False,\
            name_bigy=bigyvars1,name_bigX=bigXvars1)        

        dict_compare(reg.bOLS,{0: np.array([[ 4.50407527],[ 2.50426531],\
        [ 0.50130802]]), 1: np.array([[ 7.41171812],[ 4.0021532 ],[ 1.32168167],\
        [-0.22786048]]), 2: np.array([[ 6.92761614],[ 3.90531039],[ 1.47413939]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 4.50407527],[ 2.39199682],\
        [ 0.52723694]]), 1: np.array([[ 7.44509818],
        [ 3.74968571],[ 1.28811685],[-0.23526451]]), 2: np.array([[ 6.92761614],\
        [ 3.65423052],[ 1.38247611]])},RTOL)
        dict_compare(reg.sur_inf,{0: np.array([[  9.16019177e-002,   4.91700980e+001,   0.00000000e+000],\
        [  9.18832357e-002,   2.60330060e+001,   2.09562528e-149],\
        [  9.31668754e-002,   5.65906002e+000,   1.52204326e-008]]),\
         1: np.array([[  2.31085029e-001,   3.22180031e+001,   9.87752395e-228],\
        [  1.14421850e-001,   3.27707138e+001,   1.53941252e-235],\
        [  1.14799399e-001,   1.12205888e+001,   3.23111806e-029],\
        [  4.47806286e-002,  -5.25371170e+000,   1.49064159e-007]]),\
         2: np.array([[  1.00643767e-001,   6.88330371e+001,   0.00000000e+000],\
        [  1.00599909e-001,   3.63243917e+001,   6.66811571e-289],\
        [  1.02053898e-001,   1.35465291e+001,   8.30659234e-042]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.34470181,  0.25096458],\
       [ 0.34470181,  1.        ,  0.33527277],[ 0.25096458,  0.33527277,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.llik,-28695.767676078722)
        np.testing.assert_allclose(reg.lmtest,(882.43543942655947, 3, 5.7128374010751484e-191))
        np.testing.assert_allclose(reg.lrtest, (818.30409875688747, 3, 4.6392724270549021e-177))
        np.testing.assert_allclose(reg.lmEtest,(696.541197, 3, 1.18041989e-150))

    def test_3SLS(self): # two equations, one endog, one instrument, same k
        y_var1 = ['HR80','HR90']
        x_var1 = [['PS80','UE80'],['PS90','UE90']]
        yend_var1 = [['RD80'],['RD90']]
        q_var1 = [['FP79'],['FP89']]
        bigy1,bigX1,bigyvars1,bigXvars1 = sur_dictxy(self.db,y_var1,x_var1)
        bigyend1,bigyendvars1 = sur_dictZ(self.db,yend_var1)
        bigq1,bigqvars1 = sur_dictZ(self.db,q_var1)
        reg = ThreeSLS(bigy1,bigX1,bigyend1,bigq1)

        dict_compare(reg.b3SLS,{0: np.array([[  6.92426353e+00],[  1.42921826e+00],[  4.94348442e-04],\
        [  3.58292750e+00]]), 1: np.array([[ 7.62385875],[ 1.65031181],[-0.21682974],[ 3.91250428]])},RTOL)
        dict_compare(reg.tsls_inf,{0: np.array([[  2.32208525e-001,   2.98191616e+001,   2.20522747e-195],\
        [  1.03734166e-001,   1.37777004e+001,   3.47155373e-043],\
        [  3.08619277e-002,   1.60180675e-002,   9.87219978e-001],\
        [  1.11319989e-001,   3.21858412e+001,   2.78527634e-227]]),\
         1: np.array([[  2.87394149e-001,   2.65275364e+001,   4.66554915e-155],\
        [  9.59703138e-002,   1.71960655e+001,   2.84185085e-066],\
        [  4.08954707e-002,  -5.30204786e+000,   1.14510807e-007],\
        [  1.35867887e-001,   2.87963872e+001,   2.38043782e-182]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.26404959],
        [ 0.26404959,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(4.398001850528483, 1, 0.035981064325265613),\
        (3.3042403886525147, 1, 0.069101286634542139),\
        (21.712902666281863, 1, 3.1665430446850281e-06),\
        (4.4286185200127388, 1, 0.035341101907069621)],RTOL)

    def test_3SLS_uneqK(self): # Three equations, unequal K, two endog variables, three instruments
        y_var2 = ['HR60','HR70','HR80']
        x_var2 = [['RD60','PS60'],['RD70','PS70','MA70'],['RD80','PS80']]
        yend_var2 = [['UE60','DV60'],['UE70','DV70'],['UE80','DV80']]
        q_var2 = [['FH60','FP59','GI59'],['FH70','FP69','GI69'],['FH80','FP79','GI79']]
        bigy2,bigX2,bigyvars2,bigXvars2 = sur_dictxy(self.db,y_var2,x_var2)
        bigyend2,bigyendvars2 = sur_dictZ(self.db,yend_var2)
        bigq2,bigqvars2 = sur_dictZ(self.db,q_var2)
        reg = ThreeSLS(bigy2,bigX2,bigyend2,bigq2,name_bigy=bigyvars2,\
               name_bigX=bigXvars2,name_bigyend=bigyendvars2,\
               name_bigq=bigqvars2,name_ds="natregimes")

        dict_compare(reg.b2SLS,{0: np.array([[-2.04160355],[ 4.5438992 ],[ 1.65007567],[-0.73163458],\
        [ 5.43071683]]), 1: np.array([[ 17.26252005],[  5.17297895],[  1.2893243 ],[ -0.38349609],\
        [ -2.17689289],[  4.31713382]]), 2: np.array([[-7.6809159 ],[ 3.88957396],[ 0.49973258],\
        [ 0.36476446],[ 2.63375234]])},RTOL)
        dict_compare(reg.b3SLS,{0: np.array([[-1.56830297],[ 4.07805179],[ 1.49694849],[-0.5376807 ],\
        [ 4.65487154]]), 1: np.array([[ 16.13792395],[  4.97265632],[  1.31962844],[ -0.32122485],\
        [ -2.12407425],[  3.91227737]]), 2: np.array([[-6.7283657 ],[ 3.79206731],[ 0.52278922],\
        [ 0.33447996],[ 2.47158609]])},RTOL)
        dict_compare(reg.tsls_inf,{0: np.array([[  9.95215966e-01,  -1.57584185e+00,   1.15062254e-01],\
        [  2.26574971e-01,   1.79986861e+01,   1.99495587e-72],\
        [  1.60939740e-01,   9.30129807e+00,   1.38741353e-20],\
        [  1.19040839e-01,  -4.51677511e+00,   6.27885257e-06],\
        [  5.32942876e-01,   8.73427857e+00,   2.45216107e-18]]),\
         1: np.array([[  1.59523920e+000,   1.01163035e+001,   4.67748637e-024],\
        [  1.87013008e-001,   2.65898954e+001,   8.88419907e-156],\
        [  1.44410869e-001,   9.13801331e+000,   6.36101069e-020],\
        [  3.46429228e-002,  -9.27245233e+000,   1.81914372e-020],\
        [  2.49627824e-001,  -8.50896434e+000,   1.75493796e-017],\
        [  4.19425249e-001,   9.32771068e+000,   1.08182251e-020]]),\
         2: np.array([[  1.09143600e+000,  -6.16469102e+000,   7.06208998e-010],\
        [  1.27908896e-001,   2.96466268e+001,   3.74870055e-193],\
        [  1.32436222e-001,   3.94747912e+000,   7.89784041e-005],\
        [  8.81489692e-002,   3.79448524e+000,   1.47950082e-004],\
        [  1.95538678e-001,   1.26398834e+001,   1.27242486e-036]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.        ,  0.31819323,  0.20428789],\
       [ 0.31819323,  1.        ,  0.12492191],[ 0.20428789,  0.12492191,  1.        ]]),RTOL)

    #"""
    def test_sur_regi(self):
        y_var0 = ['HR80','HR90']
        x_var0 = [['PS80','UE80'],['PS90','UE90']]
        bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(self.db,y_var0,x_var0)
        regi1 = int(bigy0[0].shape[0]/2)
        regi = [0]*(regi1) + [1]*(bigy0[0].shape[0]-regi1)

        bigysub,bigXsub = {},{}
        for r in bigy0.keys():
            bigysub[r] = bigy0[r][0:regi1]
            bigXsub[r] = bigX0[r][0:regi1]

        reg = SUR(bigy0,bigX0,regimes=regi,name_bigy=bigyvars0,name_bigX=bigXvars0)
        reg_sub = SUR(bigysub,bigXsub,name_bigy=bigyvars0,name_bigX=bigXvars0)

        dict_compare(reg.bOLS,{0: np.array([[ 1.87615878],
       [ 0.18966296],
       [ 0.34814587],
       [ 9.16595183],
       [ 0.82165993],
       [ 0.06343039]]), 1: np.array([[ 0.74758463],
       [ 0.72948358],
       [ 0.45993437],
       [ 4.81814289],
       [ 0.96819747],
       [ 0.55080463]])},RTOL)
        dict_compare(reg_sub.bOLS,{0: np.array([[ 1.87615878],
       [ 0.18966296],
       [ 0.34814587]]), 1: np.array([[ 0.74758463],
       [ 0.72948358],
       [ 0.45993437]])},RTOL)
        dict_compare(reg.bSUR,{0: np.array([[ 2.01116476],
       [ 0.20092017],
       [ 0.32804397],
       [ 8.73384797],
       [ 0.78145176],
       [ 0.12659106]]), 1: np.array([[ 1.74977074],
       [ 0.74734938],
       [ 0.29345176],
       [ 6.31032557],
       [ 0.91171898],
       [ 0.34665252]])},RTOL)
        dict_compare(reg_sub.bSUR,{0: np.array([[ 1.92667554],
       [ 0.19603381],
       [ 0.34065072]]), 1: np.array([[ 1.48997568],
       [ 0.74311959],
       [ 0.33661536]])},RTOL)

        dict_compare(reg.sur_inf,{0: np.array([[  3.41101914e-001,   5.89608171e+000,   3.72234709e-009],
       [  1.46263739e-001,   1.37368406e+000,   1.69539787e-001],
       [  4.50557935e-002,   7.28083871e+000,   3.31751122e-013],
       [  3.54394907e-001,   2.46443947e+001,   4.22629982e-134],
       [  1.75344503e-001,   4.45666528e+000,   8.32444268e-006],
       [  4.61236608e-002,   2.74460137e+000,   6.05844400e-003]]), 1: np.array([[  3.17850453e-01,   5.50501258e+00,   3.69141910e-08],
       [  1.36607810e-01,   5.47076618e+00,   4.48094161e-08],
       [  4.66138382e-02,   6.29537851e+00,   3.06650646e-10],
       [  3.81183966e-01,   1.65545409e+01,   1.48469542e-61],
       [  1.65046297e-01,   5.52401961e+00,   3.31330455e-08],
       [  4.80006322e-02,   7.22183237e+00,   5.12916752e-13]])},RTOL)
        dict_compare(reg_sub.sur_inf,{0: np.array([[  3.09065537e-01,   6.23387375e+00,   4.55039845e-10],
       [  1.31344830e-01,   1.49251259e+00,   1.35564820e-01],
       [  4.09281853e-02,   8.32313281e+00,   8.56683667e-17]]), 1: np.array([[  2.55486625e-01,   5.83191264e+00,   5.47956072e-09],
       [  1.08792884e-01,   6.83059002e+00,   8.45660945e-12],
       [  3.75656548e-02,   8.96072109e+00,   3.22561992e-19]])},RTOL)
        np.testing.assert_allclose(reg.corr,np.array([[ 1.,         0.39876159],
 [ 0.39876159,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg_sub.corr,np.array([[ 1.,         0.34082746],
 [ 0.34082746,  1.        ]]),RTOL)
        np.testing.assert_allclose(reg.surchow,[(0.45990603232321264, 1, 0.49766789236262199), (12.272945563489683, 1, 0.00045957230926145726), (0.40387355647401846, 1, 0.52509554089354726), (29.703322949663928, 1, 5.0348441543660547e-08), (0.48278663488874679, 1, 0.48716278953324077), (14.458361295874431, 1, 0.00014329232472224597)],RTOL)
        np.testing.assert_allclose(reg_sub.surchow,[(1.6159328442921959, 1, 0.2036598282347839), (15.367000078470731, 1, 8.8520850179747918e-05), (0.0070481637293965593, 1, 0.93309352797583178)],RTOL)

    #"""

if __name__ == '__main__':
    unittest.main()
    '''
    db = pysal.open(pysal.examples.get_path('NAT.dbf'),'r')
    w = pysal.queen_from_shapefile(pysal.examples.get_path("NAT.shp"))
    w.transform = 'r'
    y_var0 = ['HR80','HR90']
    x_var0 = [['PS80','UE80'],['PS90','UE90']]
    bigy0,bigX0,bigyvars0,bigXvars0 = sur_dictxy(db,y_var0,x_var0)
    regi1 = bigy0[0].shape[0]/2
    regi = [0]*(regi1) + [1]*(bigy0[0].shape[0]-regi1)
    bigysub,bigXsub = {},{}
    for r in bigy0.keys():
        bigysub[r] = bigy0[r][0:regi1]
        bigXsub[r] = bigX0[r][0:regi1]
    reg = SUR(bigysub,bigXsub,name_bigy=bigyvars0,name_bigX=bigXvars0)
    #'''
