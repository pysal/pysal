import pysal.spreg as EC
from pysal.spreg import sputils as spu
from warnings import warn as Warn
import unittest as ut
import numpy as np
import scipy.sparse as spar

ALL_FUNCS = [f for f,v in spu.__dict__.items() \
	     if (callable(v) \
		 and not f.startswith('_'))]
COVERAGE = ['spinv', 'splogdet', 'spisfinite', 'spmin', 'spfill_diagonal', \
	    'spmax', 'spbroadcast', 'sphstack', 'spmultiply', 'spdot']

NOT_COVERED = set(ALL_FUNCS).difference(COVERAGE)

if len(NOT_COVERED) > 0:
    Warn('The following functions in {} are not covered:\n'
         '{}'.format(spu.__file__, NOT_COVERED))

class Test_Sparse_Utils(ut.TestCase):
    def setUp(self):
        np.random.seed(8879)

        self.n = 20
        self.dense0 = np.random.randint(2, size=(self.n,self.n))
        self.d0td0 = self.dense0.T.dot(self.dense0)
        self.dense1 = np.eye(self.n)
        self.sparse0 = spar.csc_matrix(self.dense0)
        self.s0ts0 = self.sparse0.T.dot(self.sparse0)
        self.sparse1 = spar.csc_matrix(spar.identity(self.n))

    def test_inv(self):
        r = spu.spinv(self.d0td0)
        rs = spu.spinv(self.s0ts0)
        rs2d = rs.toarray()

        self.assertIsInstance(r, np.ndarray)
        self.assertTrue(spar.issparse(rs))
        self.assertIsInstance(rs2d, np.ndarray)

        np.testing.assert_allclose(r, rs2d)
    
    def test_spdot(self):
        dd = spu.spdot(self.dense0, self.dense1)
        ds = spu.spdot(self.dense0, self.sparse1)
        sd = spu.spdot(self.sparse0, self.dense1)
        ss = spu.spdot(self.sparse0, self.sparse1, array_out=False)

        # typing tests
        self.assertIsInstance(dd, np.ndarray)
        self.assertIsInstance(ds, np.ndarray)
        self.assertIsInstance(sd, np.ndarray)
        self.assertIsInstance(ss, spar.csc_matrix)

        # product test
        np.testing.assert_array_equal(dd, ds)
        np.testing.assert_array_equal(dd, sd)
        np.testing.assert_array_equal(dd, ss.toarray())

    def test_logdet(self):
        dld = spu.splogdet(self.d0td0)
        sld = spu.splogdet(self.s0ts0)

        np.testing.assert_allclose(dld, sld)
   
    def test_isfinite(self):
        self.assertTrue(spu.spisfinite(self.dense0))
        self.assertTrue(spu.spisfinite(self.sparse0))
        
        dense_inf = np.float64(self.dense0.copy())
        dense_inf[0,0] = np.nan
        sparse_inf = spar.csc_matrix(dense_inf)

        self.assertTrue(not spu.spisfinite(dense_inf))
        self.assertTrue(not spu.spisfinite(sparse_inf))

    def test_min(self):
        self.assertEquals(spu.spmin(self.dense0), 0)
        self.assertEquals(spu.spmin(self.sparse0), 0)

    def test_max(self):
        self.assertEquals(spu.spmax(self.dense1), 1)
        self.assertEquals(spu.spmax(self.sparse1), 1)
    
    def test_fill_diagonal(self):
        current_dsum = self.dense0.trace()
        current_ssum = self.sparse0.diagonal().sum()
        self.assertEquals(current_dsum, 7)
        self.assertEquals(current_ssum, 7)

        tmpd = self.dense0.copy()
        tmps = self.sparse0.copy()
        d_4diag = spu.spfill_diagonal(tmpd, 4)
        s_4diag = spu.spfill_diagonal(tmps, 4)
        
        known = 4 * self.n

        self.assertEquals(known, d_4diag.trace())
        self.assertEquals(known, s_4diag.diagonal().sum())

    def test_broadcast(self):
        test_vec = np.ones((self.n,1)) * .2
        
        tmpd = spu.spbroadcast(self.dense0, test_vec)
        tmps = spu.spbroadcast(self.sparse0.tocsr(), test_vec) 

        self.assertIsInstance(tmpd, np.ndarray)
        self.assertIsInstance(tmps, spar.csr_matrix)

        np.testing.assert_allclose(tmpd, tmps.toarray())

    def test_hstack(self):
        tmpd = spu.sphstack(self.dense0, self.dense1)
        tmps = spu.sphstack(self.sparse0.tocsr(), self.sparse1.tocsr())

        self.assertIsInstance(tmpd, np.ndarray)
        self.assertIsInstance(tmps, spar.csr_matrix)

        np.testing.assert_allclose(tmpd, tmps.toarray())

    def test_multiply(self):
        dd = spu.spmultiply(self.dense0, self.dense1)
        ss = spu.spmultiply(self.sparse0, self.sparse1, array_out=False)

        #typing
        self.assertIsInstance(dd, np.ndarray)
        self.assertIsInstance(ss, spar.csc_matrix)

        #equality
        np.testing.assert_array_equal(dd, ss.toarray())


if __name__ == '__main__':
    ut.main()
