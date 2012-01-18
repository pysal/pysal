import unittest

class TestBaseGMError(unittest.TestCase):
    def test___init__(self):
        # base_g_m__error = BaseGM_Error(y, x, w)
        assert False # TODO: implement your test here

class TestGMError(unittest.TestCase):
    def test___init__(self):
        # g_m__error = GM_Error(y, x, w, vm, name_y, name_x, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMEndogError(unittest.TestCase):
    def test___init__(self):
        # base_g_m__endog__error = BaseGM_Endog_Error(y, x, yend, q, w)
        assert False # TODO: implement your test here

class TestGMEndogError(unittest.TestCase):
    def test___init__(self):
        # g_m__endog__error = GM_Endog_Error(y, x, yend, q, w, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

class TestBaseGMCombo(unittest.TestCase):
    def test___init__(self):
        # base_g_m__combo = BaseGM_Combo(y, x, yend, q, w, w_lags, lag_q)
        assert False # TODO: implement your test here

class TestGMCombo(unittest.TestCase):
    def test___init__(self):
        # g_m__combo = GM_Combo(y, x, yend, q, w, w_lags, lag_q, vm, name_y, name_x, name_yend, name_q, name_w, name_ds)
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
