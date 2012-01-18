import unittest

class TestLMtests(unittest.TestCase):
    def test___init__(self):
        # l_mtests = LMtests(ols, w, tests)
        assert False # TODO: implement your test here

class TestMoranRes(unittest.TestCase):
    def test___init__(self):
        # moran_res = MoranRes(ols, w, z)
        assert False # TODO: implement your test here

class TestAKtest(unittest.TestCase):
    def test___init__(self):
        # a_ktest = AKtest(iv, w, case)
        assert False # TODO: implement your test here

class TestSpDcache(unittest.TestCase):
    def TestAB(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.AB())
        assert False # TODO: implement your test here

    def test___init__(self):
        # sp_dcache = spDcache(reg, w)
        assert False # TODO: implement your test here

    def test_j(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.j())
        assert False # TODO: implement your test here

    def test_t(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.t())
        assert False # TODO: implement your test here

    def test_trA(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.trA())
        assert False # TODO: implement your test here

    def test_utwuDs(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.utwuDs())
        assert False # TODO: implement your test here

    def test_utwyDs(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.utwyDs())
        assert False # TODO: implement your test here

    def test_wu(self):
        # sp_dcache = spDcache(reg, w)
        # self.assertEqual(expected, sp_dcache.wu())
        assert False # TODO: implement your test here

class TestLmErr(unittest.TestCase):
    def test_lm_err(self):
        # self.assertEqual(expected, lmErr(reg, w, spDcache))
        assert False # TODO: implement your test here

class TestLmLag(unittest.TestCase):
    def test_lm_lag(self):
        # self.assertEqual(expected, lmLag(ols, w, spDcache))
        assert False # TODO: implement your test here

class TestRlmErr(unittest.TestCase):
    def test_rlm_err(self):
        # self.assertEqual(expected, rlmErr(ols, w, spDcache))
        assert False # TODO: implement your test here

class TestRlmLag(unittest.TestCase):
    def test_rlm_lag(self):
        # self.assertEqual(expected, rlmLag(ols, w, spDcache))
        assert False # TODO: implement your test here

class TestLmSarma(unittest.TestCase):
    def test_lm_sarma(self):
        # self.assertEqual(expected, lmSarma(ols, w, spDcache))
        assert False # TODO: implement your test here

class TestGetMI(unittest.TestCase):
    def test_get_m_i(self):
        # self.assertEqual(expected, get_mI(reg, w, spDcache))
        assert False # TODO: implement your test here

class TestGetVI(unittest.TestCase):
    def test_get_v_i(self):
        # self.assertEqual(expected, get_vI(ols, w, ei, spDcache))
        assert False # TODO: implement your test here

class TestGetEI(unittest.TestCase):
    def test_get_e_i(self):
        # self.assertEqual(expected, get_eI(ols, w, spDcache))
        assert False # TODO: implement your test here

class TestGetZI(unittest.TestCase):
    def test_get_z_i(self):
        # self.assertEqual(expected, get_zI(I, ei, vi))
        assert False # TODO: implement your test here

class TestAkTest(unittest.TestCase):
    def test_ak_test(self):
        # self.assertEqual(expected, akTest(iv, w, spDcache))
        assert False # TODO: implement your test here

if __name__ == '__main__':
    unittest.main()
