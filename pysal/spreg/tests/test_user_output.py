import unittest

class TestDiagnosticBuilder(unittest.TestCase):
    def test___init__(self):
        # diagnostic_builder = DiagnosticBuilder(w, vm, instruments, beta_diag, nonspat_diag, spat_diag, lamb, moran, std_err, ols, spatial_lag)
        assert False # TODO: implement your test here

class TestSetNameDs(unittest.TestCase):
    def test_set_name_ds(self):
        # self.assertEqual(expected, set_name_ds(name_ds))
        assert False # TODO: implement your test here

class TestSetNameY(unittest.TestCase):
    def test_set_name_y(self):
        # self.assertEqual(expected, set_name_y(name_y))
        assert False # TODO: implement your test here

class TestSetNameX(unittest.TestCase):
    def test_set_name_x(self):
        # self.assertEqual(expected, set_name_x(name_x, x))
        assert False # TODO: implement your test here

class TestSetNameYend(unittest.TestCase):
    def test_set_name_yend(self):
        # self.assertEqual(expected, set_name_yend(name_yend, yend))
        assert False # TODO: implement your test here

class TestSetNameQ(unittest.TestCase):
    def test_set_name_q(self):
        # self.assertEqual(expected, set_name_q(name_q, q))
        assert False # TODO: implement your test here

class TestSetNameYendSp(unittest.TestCase):
    def test_set_name_yend_sp(self):
        # self.assertEqual(expected, set_name_yend_sp(name_y))
        assert False # TODO: implement your test here

class TestSetNameQSp(unittest.TestCase):
    def test_set_name_q_sp(self):
        # self.assertEqual(expected, set_name_q_sp(name_x, w_lags, name_q, lag_q))
        assert False # TODO: implement your test here

class TestSetNameH(unittest.TestCase):
    def test_set_name_h(self):
        # self.assertEqual(expected, set_name_h(name_x, name_q))
        assert False # TODO: implement your test here

class TestSetRobust(unittest.TestCase):
    def test_set_robust(self):
        # self.assertEqual(expected, set_robust(robust))
        assert False # TODO: implement your test here

class TestSetNameW(unittest.TestCase):
    def test_set_name_w(self):
        # self.assertEqual(expected, set_name_w(name_w, w))
        assert False # TODO: implement your test here

class TestCheckArrays(unittest.TestCase):
    def test_check_arrays(self):
        # self.assertEqual(expected, check_arrays(*arrays))
        assert False # TODO: implement your test here

class TestCheckWeights(unittest.TestCase):
    def test_check_weights(self):
        # self.assertEqual(expected, check_weights(w, y))
        assert False # TODO: implement your test here

class TestCheckRobust(unittest.TestCase):
    def test_check_robust(self):
        # self.assertEqual(expected, check_robust(robust, wk))
        assert False # TODO: implement your test here

class TestCheckSpatDiag(unittest.TestCase):
    def test_check_spat_diag(self):
        # self.assertEqual(expected, check_spat_diag(spat_diag, w))
        assert False # TODO: implement your test here

class TestCheckConstant(unittest.TestCase):
    def test_check_constant(self):
        # self.assertEqual(expected, check_constant(x))
        assert False # TODO: implement your test here

class TestSummaryIntro(unittest.TestCase):
    def test_summary_intro(self):
        # self.assertEqual(expected, summary_intro(reg))
        assert False # TODO: implement your test here

class TestSummaryCoefs(unittest.TestCase):
    def test_summary_coefs(self):
        # self.assertEqual(expected, summary_coefs(reg, instruments, lamb, std_err, ols))
        assert False # TODO: implement your test here

class TestSummaryR2(unittest.TestCase):
    def test_summary_r2(self):
        # self.assertEqual(expected, summary_r2(reg, ols, spatial_lag))
        assert False # TODO: implement your test here

class TestSummaryNonspatDiag1(unittest.TestCase):
    def test_summary_nonspat_diag_1(self):
        # self.assertEqual(expected, summary_nonspat_diag_1(reg))
        assert False # TODO: implement your test here

class TestSummaryNonspatDiag2(unittest.TestCase):
    def test_summary_nonspat_diag_2(self):
        # self.assertEqual(expected, summary_nonspat_diag_2(reg))
        assert False # TODO: implement your test here

class TestSummarySpatDiag(unittest.TestCase):
    def test_summary_spat_diag(self):
        # self.assertEqual(expected, summary_spat_diag(reg, instruments, moran))
        assert False # TODO: implement your test here

class TestSummaryVm(unittest.TestCase):
    def test_summary_vm(self):
        # self.assertEqual(expected, summary_vm(reg, instruments))
        assert False # TODO: implement your test here

class TestSummaryPred(unittest.TestCase):
    def test_summary_pred(self):
        # self.assertEqual(expected, summary_pred(reg))
        assert False # TODO: implement your test here

class TestSummaryClose(unittest.TestCase):
    def test_summary_close(self):
        # self.assertEqual(expected, summary_close())
        assert False # TODO: implement your test here

class TestSummaryUnclose(unittest.TestCase):
    def test_summary_unclose(self):
        # self.assertEqual(expected, summary_unclose(summary))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
