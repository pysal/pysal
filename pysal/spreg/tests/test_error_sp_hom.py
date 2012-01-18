import unittest

class TestBaseGMErrorHom(unittest.TestCase):
    def test___init__(self):
        # base_g_m__error__hom = BaseGM_Error_Hom(y, x, w, max_iter, epsilon, A1)
        assert False # TODO: implement your test here

class TestGMErrorHom(unittest.TestCase):
    def test___init__(self):
        # g_m__error__hom = GM_Error_Hom(y, x, w, max_iter, epsilon, A1, vm, name_y, name_x, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMEndogErrorHom(unittest.TestCase):
    def test___init__(self):
        # base_g_m__endog__error__hom = BaseGM_Endog_Error_Hom(y, x, yend, q, w, constant, max_iter, epsilon, A1)
        assert False # TODO: implement your test here

class TestGMEndogErrorHom(unittest.TestCase):
    def test___init__(self):
        # g_m__endog__error__hom = GM_Endog_Error_Hom(y, x, yend, q, w, max_iter, epsilon, A1, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMComboHom(unittest.TestCase):
    def test___init__(self):
        # base_g_m__combo__hom = BaseGM_Combo_Hom(y, x, yend, q, w, w_lags, lag_q, max_iter, epsilon, A1)
        assert False # TODO: implement your test here

class TestGMComboHom(unittest.TestCase):
    def test___init__(self):
        # g_m__combo__hom = GM_Combo_Hom(y, x, yend, q, w, w_lags, lag_q, max_iter, epsilon, A1, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

class TestMomentsHom(unittest.TestCase):
    def test_moments_hom(self):
        # self.assertEqual(expected, moments_hom(w, u))
        assert False # TODO: implement your test here

class TestGetVcHom(unittest.TestCase):
    def test_get_vc_hom(self):
        # self.assertEqual(expected, get_vc_hom(w, reg, lambdapar, z_s, for_omegaOLS))
        assert False # TODO: implement your test here

class TestGetOmegaHom(unittest.TestCase):
    def test_get_omega_hom(self):
        # self.assertEqual(expected, get_omega_hom(w, reg, lamb, G))
        assert False # TODO: implement your test here

class TestGetOmegaHomOls(unittest.TestCase):
    def test_get_omega_hom_ols(self):
        # self.assertEqual(expected, get_omega_hom_ols(w, reg, lamb, G))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
