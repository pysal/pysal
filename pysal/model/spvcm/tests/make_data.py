from spvcm.utils import south
import spvcm.api as spvcm
from spvcm._constants import TEST_SEED
from spvcm.diagnostics import psrf, mcse, hpd_interval, effective_size, geweke
import numpy as np
import os

FULL_PATH = os.path.dirname(os.path.abspath(__file__))


def build():
    data = south()
    del data['W']
    del data['M']
    model = spvcm.both.MVCM(**data, n_samples=0)
    np.random.seed(TEST_SEED)
    print('starting South 5000, njobs=4')
    model.sample(5000,n_jobs=4)
    print('starting PSRF')
    known_brooks = psrf(model)
    known_gr = psrf(model, method='original')


    import json
    with open(FULL_PATH + '/data/' + 'psrf_brooks.json', 'w') as brooks:
        json.dump(known_brooks, brooks)
    with open(FULL_PATH + '/data/' + 'psrf_gr.json', 'w') as gr:
        json.dump(known_gr, gr)
    for i, method in enumerate(['bm', 'obm', 'bartlett', 'hanning']):
        known_mcse = mcse(model, varnames=['Tau2'], method=method)
        with open(FULL_PATH + '/data/' + 'mcse_{}.json'.format(i,method), 'w') as mcse_file:
            json.dump(known_mcse, mcse_file)

    known_hpd = hpd_interval(model, varnames=['Sigma2'])
    with open(FULL_PATH + '/data/' + 'hpd_interval.json', 'w') as hpd_file:
        json.dump(known_hpd, hpd_file)

    known_size = effective_size(model, varnames=['Tau2'], use_R=False)
    with open(FULL_PATH + '/data/' + 'effective_size.json', 'w') as size_file:
        json.dump(known_size, size_file)

    known_geweke = geweke(model, varnames=['Sigma2'])
    known_geweke = [{k:v.tolist() for k,v in known.items()} for known in known_geweke]
    with open(FULL_PATH + '/data/' + 'geweke.json', 'w') as geweke_file:
        json.dump(known_geweke, geweke_file)

    model.trace.to_csv(FULL_PATH + '/data/' + 'south_mvcm_5000.csv')
    return ([FULL_PATH + '/data/' + 'psrf_{}.json'.format(k)
             for k in ['brooks', 'gr']] + [FULL_PATH + '/data/south_mvcm_5000.csv'])
