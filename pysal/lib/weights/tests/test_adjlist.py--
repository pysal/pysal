import unittest as ut
import numpy as np
from .. import adjtools as adj
from ... import weights
from ... import io
from ... import examples
from ..util import lat2W
from ...common import RTOL, ATOL
from ... import io
from ...examples import get_path

try:
    import pandas
    import geopandas
    PANDAS_MISSING = False
except ImportError:
    PANDAS_MISSING = True

@ut.skipIf(PANDAS_MISSING, 'Pandas is gone')
class Test_Adjlist(ut.TestCase):
    def setUp(self):
        self.knownW = io.open(examples.get_path('columbus.gal')).read()
    
    def test_round_trip(self):
        adjlist = self.knownW.to_adjlist(remove_symmetric=False).astype(int)
        w_from_adj = weights.W.from_adjlist(adjlist)
        np.testing.assert_allclose(w_from_adj.sparse.toarray(), 
                                   self.knownW.sparse.toarray())

    def test_filter(self):
        grid = lat2W(2,2)
        alist = grid.to_adjlist(remove_symmetric=True)
        assert len(alist) == 4
        with self.assertRaises(AssertionError):
            badgrid = weights.W.from_adjlist(alist)
            np.testing.assert_allclose(badgrid.sparse.toarray(), 
                                       grid.sparse.toarray())
        assert set(alist.focal.unique().tolist()) == set(list(range(4)))
        assert set(alist.neighbor.unique().tolist()) == set(list(range(4)))
        assert alist.weight.unique().item() == 1

    def apply_and_compare_columbus(self, col):
        import geopandas
        df = geopandas.read_file(examples.get_path('columbus.dbf')).head()
        W = weights.Queen.from_dataframe(df)
        alist = adj.adjlist_apply(df[col], W=W)
        right_hovals = alist.groupby('focal').att_focal.unique()
        assert (right_hovals == df[col]).all()
        allpairs = np.subtract.outer(df[col], df[col])
        flat_diffs = allpairs[W.sparse.toarray().astype(bool)]
        np.testing.assert_allclose(flat_diffs, alist['subtract'].values)
        return flat_diffs

    def test_apply(self):
        self.apply_and_compare_columbus('HOVAL')

    def test_mvapply(self):
        import geopandas
        df = geopandas.read_file(examples.get_path('columbus.dbf')).head()
        W = weights.Queen.from_dataframe(df)
        ssq = lambda x_y: np.sum((x_y[0]-x_y[1])**2).item()
        ssq.__name__ = 'sum_of_squares'
        alist = adj.adjlist_apply(df[['HOVAL', 'CRIME', 'INC']], W=W, 
                                  func=ssq)
        known_ssq = [1301.1639302990804,
                     3163.46450914361,
                     1301.1639302990804,
                     499.52656498472993,
                     594.518273032036,
                     3163.46450914361,
                     499.52656498472993,
                     181.79100173844196,
                     436.09336916344097,
                     594.518273032036,
                     181.79100173844196,
                     481.89443401250094,
                     436.09336916344097,
                     481.89443401250094] #ugh I hate doing this, but how else?
        np.testing.assert_allclose(alist.sum_of_squares.values,
                                   np.asarray(known_ssq),
                                   rtol=RTOL, atol=ATOL)

    def test_map(self):
        atts = ['HOVAL', 'CRIME', 'INC'] 
        df = geopandas.read_file(examples.get_path('columbus.dbf')).head()
        W = weights.Queen.from_dataframe(df)
        hoval, crime, inc = list(map(self.apply_and_compare_columbus, atts)) 
        mapped = adj.adjlist_map(df[atts], W=W)
        for name,data in zip(atts, (hoval, crime, inc)):
            np.testing.assert_allclose(data, 
                                       mapped['_'.join(('subtract',name))].values)
