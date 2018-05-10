from spvcm._constants import TEST_SEED, CLASSTYPES
from ... import both_levels as M
from spvcm.tests.utils import run_with_seed
from spvcm.abstracts import Sampler_Mixin
import pandas as pd
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def build():
    models = []
    for cand in M.__dict__.values():
        if isinstance(cand, CLASSTYPES):
            if issubclass(cand, Sampler_Mixin):
                models.append(cand)
    for model in models:
        print('starting {}'.format(model))
        run_with_seed(model, seed=TEST_SEED, fprefix=FULL_PATH + '/data/')
    return os.listdir(FULL_PATH + '/data/')
