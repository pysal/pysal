import unittest

class TestBaseGMErrorHet(unittest.TestCase):
    def test___init__(self):
        # base_g_m__error__het = BaseGM_Error_Het(y, x, w, max_iter, epsilon, step1c)
        assert False # TODO: implement your test here

class TestGMErrorHet(unittest.TestCase):
    def test___init__(self):
        # g_m__error__het = GM_Error_Het(y, x, w, max_iter, epsilon, step1c, vm, name_y, name_x, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMEndogErrorHet(unittest.TestCase):
    def test___init__(self):
        # base_g_m__endog__error__het = BaseGM_Endog_Error_Het(y, x, yend, q, w, constant, max_iter, epsilon, step1c, inv_method)
        assert False # TODO: implement your test here

class TestGMEndogErrorHet(unittest.TestCase):
    def test___init__(self):
        # g_m__endog__error__het = GM_Endog_Error_Het(y, x, yend, q, w, max_iter, epsilon, step1c, inv_method, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMComboHet(unittest.TestCase):
    def test___init__(self):
        # base_g_m__combo__het = BaseGM_Combo_Het(y, x, yend, q, w, w_lags, lag_q, max_iter, epsilon, step1c, inv_method)
        assert False # TODO: implement your test here

class TestGMComboHet(unittest.TestCase):
    def test___init__(self):
        # g_m__combo__het = GM_Combo_Het(y, x, yend, q, w, w_lags, lag_q, max_iter, epsilon, step1c, inv_method, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

class TestGetPsiSigma(unittest.TestCase):
    def test_get_psi_sigma(self):
        # self.assertEqual(expected, get_psi_sigma(w, u, lamb))
        assert False # TODO: implement your test here

class TestGetVcHet(unittest.TestCase):
    def test_get_vc_het(self):
        # self.assertEqual(expected, get_vc_het(w, E))
        assert False # TODO: implement your test here

class TestGetVmHet(unittest.TestCase):
    def test_get_vm_het(self):
        # self.assertEqual(expected, get_vm_het(G, lamb, reg, w, psi))
        assert False # TODO: implement your test here

class TestGetPHat(unittest.TestCase):
    def test_get__p_hat(self):
        # self.assertEqual(expected, get_P_hat(reg, hthi, zf))
        assert False # TODO: implement your test here

class TestGetA1a2(unittest.TestCase):
    def test_get_a1a2(self):
        # self.assertEqual(expected, get_a1a2(w, reg, lambdapar, P, zs, inv_method, filt))
        assert False # TODO: implement your test here

class TestGetVcHetTsls(unittest.TestCase):
    def test_get_vc_het_tsls(self):
        # self.assertEqual(expected, get_vc_het_tsls(w, reg, lambdapar, P, zs, inv_method, filt, save_a1a2))
        assert False # TODO: implement your test here

class TestGetOmegaGS2SLS(unittest.TestCase):
    def test_get__omega__g_s2_sl_s(self):
        # self.assertEqual(expected, get_Omega_GS2SLS(w, lamb, reg, G, psi, P))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
