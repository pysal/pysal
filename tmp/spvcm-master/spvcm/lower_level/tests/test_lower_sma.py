from spvcm import lower_level as lower
from spvcm.tests.utils import Model_Mixin
from spvcm.abstracts import Trace
import unittest as ut
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Lower_SMA(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_Lower_SMA, self).build_self()
        self.cls = lower.SMA
        del self.inputs["M"]
        self.inputs['n_samples'] = 0
        instance = self.cls(**self.inputs)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/lower_sma.csv')
