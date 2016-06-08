import util as ut
import pysal as ps
from warnings import warn as Warn
from pysal.weights2 import util as u

class Test_Utils(ut.TestCase):
    def setUp(self):
        all_functions = u.__all__
        self._registered_tests = []
        is_tested = [t in self._registered_tests for t in all_functions]
        untested = set(all_functions).difference(set(self.registered_tests))
        if untested != set():
            Warn("there are {} untested functions in {}."
                 " They are: {}".format(len(untested), __file__, untested))
        raise NotImplementedError
    def test_all(self):
        raise NotImplementedError
