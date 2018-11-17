from spvcm.hierarchical.svc import SVC
from spvcm._constants import TEST_SEED
from spvcm.utils import baltim
import pysal as ps
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))

def build():
    inputs = baltim()
    inputs.update(dict(configs=dict(jump=.5)))
    model = SVC(**inputs, n_samples=0)
    np.random.seed(TEST_SEED)
    model.draw()
    model.trace.to_csv(FULL_PATH + '/data/svc.csv')
