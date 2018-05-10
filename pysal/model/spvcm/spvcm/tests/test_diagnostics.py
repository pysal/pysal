import unittest as ut
import numpy as np
from spvcm.utils import south
from spvcm.diagnostics import psrf, geweke, effective_size, hpd_interval, summarize, mcse
from spvcm._constants import RTOL, ATOL, TEST_SEED
from spvcm.abstracts import Trace, Hashmap
import os
import json
FULL_PATH = os.path.dirname(os.path.abspath(__file__))


class Test_PSRF(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        with open(FULL_PATH + '/data/psrf_noburn.json', 'r') as noburn:
            self.noburn = json.load(noburn)
        with open(FULL_PATH + '/data/psrf_brooks.json', 'r') as brooks:
            self.known_brooks = json.load(brooks)
        with open(FULL_PATH + '/data/psrf_gr.json', 'r') as gr:
            self.known_gr = json.load(gr)
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.mockmodel = Hashmap(trace=self.trace)

    def test_coef_recovery(self):
        #test with:
        #model=model, trace=model.trace, chain=model.trace['asdf']
        #autoburnin=False, method='original'
        exp_brooks = psrf(self.mockmodel)
        for k,v in exp_brooks.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, np.squeeze(self.known_brooks[k]),
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        exp_gr = psrf(trace=self.trace, method='original')
        for k,v in exp_gr.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, np.squeeze(self.known_gr[k]),
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))

    def test_options(self):
        exp_brooks = psrf(trace=self.mockmodel.trace)
        for k,v in exp_brooks.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, np.squeeze(self.known_brooks[k]),
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        exp_brooks = psrf(chain=self.mockmodel.trace['Tau2'])
        np.testing.assert_allclose(exp_brooks['parameter'],
                                   self.known_brooks['Tau2'],
                                   rtol=RTOL, atol=ATOL,
                                   err_msg='Failed in Tau2')
        test_completion = psrf(trace=self.trace, autoburnin=False)
        for k,v in self.noburn.items():
            if k == 'Alphas':
                continue
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(v, np.squeeze(test_completion[k]),
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))
        limit_vars = psrf(trace=self.trace, varnames=['Tau2', 'Sigma2'])
        for k,v in limit_vars.items():
            if k == 'Alphas':
                continue
            np.testing.assert_allclose(v, np.squeeze(limit_vars[k]),
                                       rtol=RTOL, atol=ATOL,
                                       err_msg='Failed in {}'.format(k))

class Test_Gekewe(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.single_trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000_0.csv')
        self.geweke_known = json.load(open(FULL_PATH + '/data/geweke.json'))

    def test_values(self):
        single_size = geweke(trace=self.single_trace, varnames='Sigma2')
        multi_size = geweke(trace=self.trace, varnames='Sigma2')
        np.testing.assert_allclose(single_size[0]['Sigma2'], self.geweke_known[0]['Sigma2'])
        np.testing.assert_allclose(multi_size[0]['Sigma2'], self.geweke_known[0]['Sigma2'])

class Test_Effective_Size(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.single_trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000_0.csv')
        self.size_known = json.load(open(FULL_PATH + '/data/effective_size.json', 'r'))

    def test_values(self):
        single_size = effective_size(trace=self.single_trace, use_R = False, varnames='Tau2')
        multi_size = effective_size(trace=self.trace, use_R = False, varnames='Tau2')
        np.testing.assert_allclose(single_size['Tau2'], multi_size[0]['Tau2'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(single_size['Tau2'], self.size_known[0]['Tau2'], rtol=RTOL, atol=ATOL)


class Test_HPD_Interval(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.single_trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000_0.csv')
        self.hpd_known = json.load(open(FULL_PATH + '/data/hpd_interval.json', 'r'))[0]

    def test_values(self):
        single_hpd = hpd_interval(trace=self.single_trace)
        multi_hpd = hpd_interval(trace=self.trace)
        np.testing.assert_allclose(single_hpd['Sigma2'], multi_hpd[0]['Sigma2'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(single_hpd['Sigma2'], self.hpd_known['Sigma2'], rtol=RTOL, atol=ATOL)

class Test_Summarize(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        np.random.seed(TEST_SEED)
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)

class Test_MCMCSE(ut.TestCase):
    def setUp(self):
        data = south()
        data['n_samples'] = 0
        np.random.seed(TEST_SEED)
        test_methods = ['obm', 'bm', 'bartlett', 'hanning', 'tukey']
        self.trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000', multi=True)
        self.single_trace = Trace.from_csv(FULL_PATH + '/data/south_mvcm_5000_0.csv')
        self.bm = json.load(open(FULL_PATH + '/data/mcse_bm.json', 'r'))
        self.obm = json.load(open(FULL_PATH + '/data/mcse_obm.json', 'r'))
        self.tukey = json.load(open(FULL_PATH + '/data/mcse_hanning.json', 'r'))
        self.bartlett = json.load(open(FULL_PATH + '/data/mcse_bartlett.json', 'r'))
        self.hanning = self.tukey

    def test_method_values(self):
        for method in ['bm', 'obm', 'hanning', 'bartlett', 'tukey']:
            multi_ses = mcse(trace=self.trace, varnames=['Tau2'], method=method)
            single_ses = mcse(trace=self.single_trace, varnames=['Tau2'], method=method)
            np.testing.assert_allclose(multi_ses[0]['Tau2'], single_ses['Tau2'], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(getattr(self, method)['Tau2'], single_ses['Tau2'], rtol=RTOL, atol=ATOL)

    def test_varying_scale(self):
        _  =mcse(trace=self.single_trace, varnames=['Tau2'], rescale=3)
