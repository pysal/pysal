from pysal.model.spvcm.svc import SVC 
from pysal.model.spvcm._constants import TEST_SEED
from pysal.model.spvcm.utils import baltim
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))


def build():
    inputs = baltim()
    inputs.update(dict(configs=dict(jump=.5)))
    model = SVC(**inputs, n_samples=0)
    np.random.seed(TEST_SEED)
    model.sample(1)
    model.trace.to_csv(FULL_PATH + '/data/svc.csv')


if __name__ == '__main__':
    build()
