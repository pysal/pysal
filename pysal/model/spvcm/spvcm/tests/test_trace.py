import numpy as np
import pandas as pd
from spvcm.abstracts import Hashmap, Trace
import unittest as ut
from spvcm._constants import RTOL, ATOL
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Trace(ut.TestCase):
    def setUp(self):

        self.a = {chr(i+97):list(range(10)) for i in range(5)}

        self.t = Trace(**self.a)
        self.mt = Trace(self.a,self.a,self.a)
        self.real_mt = Trace.from_csv(FULL_PATH + r'/data/south_mvcm_5000', multi=True)
        self.real_singles = [Trace.from_csv(FULL_PATH + r'/data/south_mvcm_5000_{}.csv'
                             .format(i)) for i in range(4)]

    def test_validate_names(self):
        b = self.a.copy()
        try:
            bad_names = Trace(self.a,b,self.a,self.a)
        except KeyError:
            pass

    # tests
    def test_slicing(self):
        t = self.t
        mt = self.mt
        #1index
        assert t[1] == {'a':1, 'b':1, 'c':1, 'd':1, 'e':1}
        assert mt[6] == [{k:6 for k in ['a','b', 'c', 'd', 'e']}]*3
        assert t[-1] =={k:9 for k in  ['a','b', 'c', 'd', 'e']}
        assert mt[-1] == [{k:9 for k in ['a','b', 'c', 'd', 'e']}]*3

        assert t[2:5] == {k:list(range(2,5)) for k in  ['a','b', 'c', 'd', 'e']}
        assert mt[8:] == [ {k:list(range(8,10)) for k in  ['a','b', 'c', 'd', 'e'] }] * 3
        assert t[-4::2] ==  {k:[6,8] for k in  ['a','b', 'c', 'd', 'e']}

        assert (t['a'] == list(range(10))).all()
        assert (mt['a'] == [list(range(10))]*3).all()
        assert t[['a','b']] == {'a':list(range(10)), 'b':list(range(10))}
        assert mt[['a','b']] == [{'a':list(range(10)), 'b':list(range(10))}]*3

        #2index
        assert t['a', 1] == 1
        assert t[['a', 'b'], 1] == {'a':1, 'b':1}
        assert (mt['e', 5] == [5]*3).all()
        assert mt[['d', 'e'], 8:] == [{'d':[8,9], 'e':[8,9]}]*3

        assert (t[0, 'a'] == list(range(10))).all()
        assert t[0, ['a', 'b']] == {'a':list(range(10)), 'b':list(range(10))}
        try:
            t[1, ['a','c']]
            raise Exception('This did not raise an exception within the slicer!')
        except IndexError:
            pass
        assert mt[1:, ['a','c']] == [{'a':list(range(10)), 'c':list(range(10))}] * 2
        assert (mt[2, 'a'] == list(range(10))).all()
        assert t[0,-1] == {k:9 for k in ['a', 'b', 'c', 'd', 'e']}
        assert t[0,:] ==  {k:list(range(10)) for k in  ['a', 'b', 'c', 'd', 'e']}
        assert mt[:, -1:-4:-1] ==  [{k:[9,8,7] for k in ['a', 'b', 'c', 'd', 'e']}]*3

        #3index
        assert t[0, 'a', -1] == 9
        assert t[0, ['a','b'],-3::2] == {'a':[7,9], 'b':[7,9]}
        assert t[0, : ,-1] == {k:9 for k in ['a','b','c','d','e']}
        try:
            t[1, 'a', -1]
            raise Exception('this did not raise an exception when it should have')
        except IndexError:
            pass
        assert (mt[1:, 'a', -1] == [9]*2).all()
        assert mt[1:, ['a','b'], -2:] == [{'a':[8,9], 'b':[8,9]}]*2
        assert (mt[2, 'a', 5::2] == [5,7,9]).all()
        assert (mt[1:, 'a', -5::2] == [[5,7,9]]*2).all()
        assert (mt[:, 'a', -5::2] == [[5,7,9]]*3).all()
        assert mt[2, :, :] == {k:list(range(10)) for k in ['a','b','c','d','e']}
        assert mt[:,:,:] == mt.chains
        assert mt[:,:,:] is not mt.chains

    def test_to_df(self):
        df = self.t.to_df()
        df2 = pd.DataFrame.from_dict(self.t.chains[0])
        np.testing.assert_array_equal(df.values, df2.values)
        mtdf = self.mt.to_df()
        mtdf2 = [pd.DataFrame.from_dict(chain) for chain in self.mt.chains]
        for i in range(len(mtdf2)):
            np.testing.assert_array_equal(mtdf[i].values, mtdf2[i].values)

    def test_from_df(self):
        df = self.t.to_df()
        new_trace = Trace.from_df(df)
        assert new_trace == self.t
        new_mt = Trace.from_df((df, df, df))
        assert new_mt == self.t

    def test_to_csv(self):
        df = self.t.to_df()
        self.t.to_csv('./test_to_csv.csv')
        new_df = pd.read_csv('./test_to_csv.csv')
        np.testing.assert_allclose(df.values, new_df.values,
                                    rtol=RTOL, atol=ATOL)
        os.remove('./test_to_csv.csv')

    def test_from_csv(self):
        self.t.to_csv('./test_from_csv.csv')
        new_t = Trace.from_csv('./test_from_csv.csv')
        assert self.t == new_t
        os.remove('./test_from_csv.csv')

    def test_single_roundtrips(self):
        source_from_file = self.real_singles[0]
        from_df = Trace.from_df(source_from_file.to_df())
        source_from_file._assert_allclose(from_df)

    def test_ordering(self):
        for ch, alone in zip(self.real_mt.chains, self.real_singles):
            Trace(ch)._assert_allclose(alone)

    def test_multi_roundtrips(self):
        dfs = self.real_mt.to_df()
        new = Trace.from_df(dfs)
        new._assert_allclose(self.real_mt)


    @ut.skip
    def test_from_pymc3(self):
        raise NotImplementedError

    @ut.skip
    def test_plot(self):
        try:
            import matplotlib as mpl
            mpl.use('Agg')
            self.t.plot()
        except:
            raise Exception('Single trace plotting failed!')

        try:
            import matplotlib as mpl
            mpl.use('Agg')
            self.mt.plot()
        except:
            raise Exception('Multi-chain trace plotting failed!')
