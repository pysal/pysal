from spvcm import both_levels as M
from spvcm.tests.utils import Model_Mixin
from spvcm.abstracts import Trace
import unittest as ut
import pandas as pd
from .make_data import FULL_PATH

class Test_SMASMA(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_SMASMA, self).build_self()
        self.cls = M.SMASMA
        self.inputs['n_samples'] = 0
        self.instance = self.cls(**self.inputs)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/smasma.csv')
