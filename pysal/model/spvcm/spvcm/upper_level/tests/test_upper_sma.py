from spvcm import upper_level as upper
from spvcm import utils
from spvcm.tests.utils import Model_Mixin
from spvcm.abstracts import Trace
import unittest as ut
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

class Test_Upper_SMA(ut.TestCase, Model_Mixin):
    def setUp(self):
        super(Test_Upper_SMA, self).build_self()
        self.cls = upper.SMA
        del self.inputs["W"]
        self.inputs['n_samples'] = 0
        instance = self.cls(**self.inputs)
        self.answer_trace = Trace.from_csv(FULL_PATH + '/data/upper_sma.csv')
