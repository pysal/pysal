from spvcm import both_levels as M
from spvcm.tests.utils import Model_Mixin
from spvcm.abstracts import Trace
import unittest as ut
import pandas as pd
from .make_data import FULL_PATH

class Test_MVCM(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_MVCM, self).build_self()
        self.cls = M.MVCM
        del self.inputs['M']
        del self.inputs['W']
        self.inputs['n_samples'] = 0
        self.instance = self.cls(**self.inputs)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/mvcm.csv')
