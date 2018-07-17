import unittest as ut
import numpy as np
from ..util import lat2W
from ..weights import W
try:
    import networkx as nx
except ImportError:
    nx = None

@ut.skipIf(nx is None, "Missing networkx")
class Test_NetworkXConverter(ut.TestCase):
    def setUp(self):
        self.known_nx = nx.random_regular_graph(4,10,seed=8879) 
        self.known_amat = np.array(nx.to_numpy_matrix(self.known_nx))
        self.known_W = lat2W(5,5)

    def test_round_trip(self):
        W_ = W.from_networkx(self.known_nx)
        np.testing.assert_allclose(W_.sparse.toarray(), self.known_amat)
        nx2 = W_.to_networkx()
        np.testing.assert_allclose(nx.to_numpy_matrix(nx2), self.known_amat)
        nxsquare = self.known_W.to_networkx()
        np.testing.assert_allclose(self.known_W.sparse.toarray(), nx.to_numpy_matrix(nxsquare))
        W_square = W.from_networkx(nxsquare)
        np.testing.assert_allclose(self.known_W.sparse.toarray(), W_square.sparse.toarray())
