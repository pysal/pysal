import unittest as ut
from check import safe_import

class TestSafeImports(ut.TestCase):
    """
        This test makes sure we don't remove anything from the pysal NameSpace that
        1.0 users might expect to be there.  1.0 Namespace was taken from the 1.1
        Code sprint wave, with special names removes (__all__, etc)
    """
    def test_this(self):
        import this
        self.assertEqual(hash(this), hash(safe_import('this')))
    def test_one_from_scipy(self):
        from scipy import stats
        self.assertEqual(hash(stats), hash(safe_import('scipy', submods='stats')))
    def test_many_from_scipy(self):
        from scipy import stats, spatial
        submods = [hash(stats), hash(spatial)]
        test_submods = safe_import('scipy', submods=['stats', 'spatial'])
        hash_test = [hash(x) for x in test_submods]
        self.assertEqual(submods, hash_test)
    
suite = ut.TestLoader().loadTestsFromTestCase(TestSafeImports)

if __name__ == '__main__':
    ut.main()
    runner = ut.TextTestRunner()
    runner.run(suite)
