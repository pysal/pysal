import unittest

class TestRegressionPropsY(unittest.TestCase):
    def test_mean_y(self):
        # regression_props_y = RegressionPropsY()
        # self.assertEqual(expected, regression_props_y.mean_y())
        assert False # TODO: implement your test here

    def test_std_y(self):
        # regression_props_y = RegressionPropsY()
        # self.assertEqual(expected, regression_props_y.std_y())
        assert False # TODO: implement your test here

class TestRegressionPropsVM(unittest.TestCase):
    def test_sig2n(self):
        # regression_props_v_m = RegressionPropsVM()
        # self.assertEqual(expected, regression_props_v_m.sig2n())
        assert False # TODO: implement your test here

    def test_sig2n_k(self):
        # regression_props_v_m = RegressionPropsVM()
        # self.assertEqual(expected, regression_props_v_m.sig2n_k())
        assert False # TODO: implement your test here

    def test_utu(self):
        # regression_props_v_m = RegressionPropsVM()
        # self.assertEqual(expected, regression_props_v_m.utu())
        assert False # TODO: implement your test here

    def test_vm(self):
        # regression_props_v_m = RegressionPropsVM()
        # self.assertEqual(expected, regression_props_v_m.vm())
        assert False # TODO: implement your test here

class TestGetA1Het(unittest.TestCase):
    def test_get__a1_het(self):
        # self.assertEqual(expected, get_A1_het(S))
        assert False # TODO: implement your test here

class TestGetA1Hom(unittest.TestCase):
    def test_get__a1_hom(self):
        # self.assertEqual(expected, get_A1_hom(s, scalarKP))
        assert False # TODO: implement your test here

class TestGetA2Hom(unittest.TestCase):
    def test_get__a2_hom(self):
        # self.assertEqual(expected, get_A2_hom(s))
        assert False # TODO: implement your test here

class TestOptimMoments(unittest.TestCase):
    def test_optim_moments(self):
        # self.assertEqual(expected, optim_moments(moments_in, vcX))
        assert False # TODO: implement your test here

class TestFoptimPar(unittest.TestCase):
    def test_foptim_par(self):
        # self.assertEqual(expected, foptim_par(par, moments))
        assert False # TODO: implement your test here

class TestGetSpFilter(unittest.TestCase):
    def test_get_sp_filter(self):
        # self.assertEqual(expected, get_spFilter(w, lamb, sf))
        assert False # TODO: implement your test here

class TestGetLags(unittest.TestCase):
    def test_get_lags(self):
        # self.assertEqual(expected, get_lags(w, x, w_lags))
        assert False # TODO: implement your test here

class TestInverseProd(unittest.TestCase):
    def test_inverse_prod(self):
        # self.assertEqual(expected, inverse_prod(w, data, scalar, post_multiply, inv_method, threshold, max_iterations))
        assert False # TODO: implement your test here

class TestPowerExpansion(unittest.TestCase):
    def test_power_expansion(self):
        # self.assertEqual(expected, power_expansion(w, data, scalar, post_multiply, threshold, max_iterations))
        assert False # TODO: implement your test here

class TestRevLagSpatial(unittest.TestCase):
    def test_rev_lag_spatial(self):
        # self.assertEqual(expected, rev_lag_spatial(w, y))
        assert False # TODO: implement your test here

class TestSetEndog(unittest.TestCase):
    def test_set_endog(self):
        # self.assertEqual(expected, set_endog(y, x, w, yend, q, w_lags, lag_q))
        assert False # TODO: implement your test here

class TestIterMsg(unittest.TestCase):
    def test_iter_msg(self):
        # self.assertEqual(expected, iter_msg(iteration, max_iter))
        assert False # TODO: implement your test here

class TestSpAtt(unittest.TestCase):
    def test_sp_att(self):
        # self.assertEqual(expected, sp_att(w, y, predy, w_y, rho))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
