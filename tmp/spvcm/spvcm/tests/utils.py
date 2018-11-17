import unittest as ut
from spvcm import utils
from spvcm._constants import RTOL, ATOL, TEST_SEED, CLASSTYPES
from warnings import warn
import numpy as np
import copy
import types

class Model_Mixin(object):
    def build_self(self):
        super(Model_Mixin, self).__init__()
        self.inputs = utils.south()
        self.__dict__.update(self.inputs)
        self.ignore_shape = False
        self.squeeze = True

    def test_trace(self):
        self.inputs['n_samples'] = 0
        instance = self.cls(**self.inputs)
        np.random.seed(TEST_SEED)
        instance.draw()
        instance.trace._assert_allclose(self.answer_trace,
                                        rtol=RTOL, atol=ATOL,
                                        ignore_shape = self.ignore_shape,
                                        squeeze=self.squeeze)

    @ut.skip
    def test_argument_parsing(self):
        #priors, initial values, etc.
        raise NotImplementedError

    @ut.skip
    def test_covariance_assignment(self):
        raise NotImplementedError

def run_with_seed(cls, env=utils.south(), seed=TEST_SEED, fprefix = ''):
    fname = str(cls).strip("'<>'").split('.')[-1].lower()
    try:
        env['n_samples'] = 0
        model = cls(**env)
    except TypeError:
        reduced = copy.deepcopy(env)
        del reduced['M']
        del reduced['W']
        reduced['n_samples'] = 0
        model = cls(**reduced)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_df().to_csv(fprefix + fname + '.csv', index=False)
