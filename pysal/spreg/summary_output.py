"""Internal helper files for user output."""

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu, Jing Yao jingyao@asu.edu"

import textwrap as TW
import numpy as np
import copy as COPY
import diagnostics as diagnostics
import diagnostics_tsls as diagnostics_tsls
import diagnostics_sp as diagnostics_sp
import pysal
import scipy
from scipy.sparse.csr import csr_matrix

__all__ = []


###############################################################################
############### Primary functions for running summary diagnostics #############
###############################################################################

"""
This section contains one function for each user level regression class. These
are called directly from the user class. Each one mixes and matches smaller
functions located later in this module.
"""


def OLS(reg, vm, w, nonspat_diag, spat_diag, moran, white_test, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_ols(reg, reg.robust)
    if nonspat_diag:
        # compute diagnostics
        reg.sig2ML = reg.sig2n
        reg.f_stat = diagnostics.f_stat(reg)
        reg.logll = diagnostics.log_likelihood(reg)
        reg.aic = diagnostics.akaike(reg)
        reg.schwarz = diagnostics.schwarz(reg)
        reg.mulColli = diagnostics.condition_index(reg)
        reg.jarque_bera = diagnostics.jarque_bera(reg)
        reg.breusch_pagan = diagnostics.breusch_pagan(reg)
        reg.koenker_bassett = diagnostics.koenker_bassett(reg)
        if white_test:
            reg.white = diagnostics.white(reg)
        # organize summary output
        reg.__summary['summary_nonspat_diag_1'] = summary_nonspat_diag_1(reg)
        reg.__summary['summary_nonspat_diag_2'] = summary_nonspat_diag_2(reg)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_ols(reg, w, moran)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=nonspat_diag, spat_diag=spat_diag)


def OLS_multi(reg, multireg, vm, nonspat_diag, spat_diag, moran, white_test, regimes=False, sur=False, w=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_ols(mreg, mreg.robust)
        if nonspat_diag:
            # compute diagnostics
            mreg.sig2ML = mreg.sig2n
            mreg.f_stat = diagnostics.f_stat(mreg)
            mreg.logll = diagnostics.log_likelihood(mreg)
            mreg.aic = diagnostics.akaike(mreg)
            mreg.schwarz = diagnostics.schwarz(mreg)
            mreg.mulColli = diagnostics.condition_index(mreg)
            mreg.jarque_bera = diagnostics.jarque_bera(mreg)
            mreg.breusch_pagan = diagnostics.breusch_pagan(mreg)
            mreg.koenker_bassett = diagnostics.koenker_bassett(mreg)
            if white_test:
                mreg.white = diagnostics.white(mreg)
            # organize summary output
            mreg.__summary[
                'summary_nonspat_diag_1'] = summary_nonspat_diag_1(mreg)
            mreg.__summary[
                'summary_nonspat_diag_2'] = summary_nonspat_diag_2(mreg)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_ols(mreg, mreg.w, moran)
        if regimes:
            summary_regimes(mreg, chow=False)
        if sur:
            summary_sur(mreg)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg)
    if sur:
        summary_sur(reg, u_cov=True)
    if spat_diag:
        # compute global diagnostics and organize summary output
        spat_diag_ols(reg, w, moran)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm, instruments=False,
                  nonspat_diag=nonspat_diag, spat_diag=spat_diag)


def TSLS(reg, vm, w, spat_diag, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, reg.robust)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_instruments(reg, w)
    # build coefficients table body
    build_coefs_body_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=spat_diag)


def TSLS_multi(reg, multireg, vm, spat_diag, regimes=False, sur=False, w=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, mreg.robust)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_instruments(mreg, mreg.w)
        # build coefficients table body
        build_coefs_body_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        if sur:
            summary_sur(mreg)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg)
    if sur:
        summary_sur(reg, u_cov=True)
    if spat_diag:
        # compute global diagnostics and organize summary output
        spat_diag_instruments(reg, w)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=spat_diag)


def GM_Lag(reg, vm, w, spat_diag, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, reg.robust, error=False)
    if spat_diag:
        # compute diagnostics and organize summary output
        spat_diag_instruments(reg, w)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=spat_diag)


def GM_Lag_multi(reg, multireg, vm, spat_diag, regimes=False, sur=False, w=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, mreg.robust, error=False)
        if spat_diag:
            # compute diagnostics and organize summary output
            spat_diag_instruments(mreg, mreg.w)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        if sur:
            summary_sur(mreg)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg)
    if spat_diag:
        pass
        # compute global diagnostics and organize summary output
        #spat_diag_instruments(reg, w)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=spat_diag)


def ML_Lag(reg, w, vm, spat_diag, regimes=False):  # extra space d
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, robust=None, error=False)
    reg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'Sigma-square ML', reg.sig2, 'Log likelihood', reg.logll)
    reg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'S.E of regression', np.sqrt(reg.sig2), 'Akaike info criterion', reg.aic)
    reg.__summary['summary_r2'] += "                                                 %-22s:%12.3f\n" % (
        'Schwarz criterion', reg.schwarz)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=False, spat_diag=spat_diag)


# extra space d
def ML_Lag_multi(reg, multireg, vm, spat_diag, regimes=False, sur=False, w=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, robust=None, error=False)
        mreg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            'Sigma-square ML', mreg.sig2, 'Log likelihood', mreg.logll)
        mreg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            'S.E of regression', np.sqrt(mreg.sig2), 'Akaike info criterion', mreg.aic)
        mreg.__summary['summary_r2'] += "                                                 %-22s:%12.3f\n" % (
            'Schwarz criterion', mreg.schwarz)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=False, nonspat_diag=False, spat_diag=spat_diag)


def ML_Error(reg, w, vm, spat_diag, regimes=False):   # extra space d
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, robust=None)
    reg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'Sigma-square ML', reg.sig2, 'Log likelihood', reg.logll)
    reg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'S.E of regression', np.sqrt(reg.sig2), 'Akaike info criterion', reg.aic)
    reg.__summary['summary_r2'] += "                                                 %-22s:%12.3f\n" % (
        'Schwarz criterion', reg.schwarz)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=False, spat_diag=spat_diag)


# extra space d
def ML_Error_multi(reg, multireg, vm, spat_diag, regimes=False, sur=False, w=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, robust=None)
        mreg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            'Sigma-square ML', mreg.sig2, 'Log likelihood', mreg.logll)
        mreg.__summary['summary_r2'] += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            'S.E of regression', np.sqrt(mreg.sig2), 'Akaike info criterion', mreg.aic)
        mreg.__summary['summary_r2'] += "                                                 %-22s:%12.3f\n" % (
            'Schwarz criterion', mreg.schwarz)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=False, nonspat_diag=False, spat_diag=spat_diag)


def GM_Error(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=False, spat_diag=False)


def GM_Error_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=False)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=False, nonspat_diag=False, spat_diag=False)


def GM_Endog_Error(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Endog_Error_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=False)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def GM_Error_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    summary_iteration(reg)
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=False, spat_diag=False)


def GM_Error_Hom_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        summary_iteration(mreg)
        beta_diag(mreg, None)
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=False, nonspat_diag=False, spat_diag=False)


def GM_Endog_Error_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    summary_iteration(reg)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Endog_Error_Hom_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, None)
        summary_iteration(mreg)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def GM_Error_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, 'het')
    summary_iteration(reg)
    # build coefficients table body
    beta_position = summary_coefs_somex(reg, reg.z_stat)
    summary_coefs_lambda(reg, reg.z_stat)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=False,
            nonspat_diag=False, spat_diag=False)


def GM_Error_Het_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, 'het')
        summary_iteration(mreg)
        # build coefficients table body
        beta_position = summary_coefs_somex(mreg, mreg.z_stat)
        summary_coefs_lambda(mreg, mreg.z_stat)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=False, nonspat_diag=False, spat_diag=False)


def GM_Endog_Error_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, 'het')
    summary_iteration(reg)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Endog_Error_Het_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag(mreg, 'het')
        summary_iteration(mreg)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def GM_Combo(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, None)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    summary_warning(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Combo_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, None)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg, lambd=False)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def GM_Combo_Hom(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, None)
    summary_iteration(reg)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Combo_Hom_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, None)
        summary_iteration(mreg)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def GM_Combo_Het(reg, vm, w, regimes=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag_lag(reg, 'het')
    summary_iteration(reg)
    # build coefficients table body
    summary_coefs_allx(reg, reg.z_stat, lambd=True)
    summary_coefs_lambda(reg, reg.z_stat)
    summary_coefs_instruments(reg)
    if regimes:
        summary_regimes(reg)
    summary_warning(reg)
    summary(reg=reg, vm=vm, instruments=True,
            nonspat_diag=False, spat_diag=False)


def GM_Combo_Het_multi(reg, multireg, vm, regimes=False):
    for m in multireg:
        mreg = multireg[m]
        mreg.__summary = {}
        # compute diagnostics and organize summary output
        beta_diag_lag(mreg, 'het')
        summary_iteration(mreg)
        # build coefficients table body
        summary_coefs_allx(mreg, mreg.z_stat, lambd=True)
        summary_coefs_lambda(mreg, mreg.z_stat)
        summary_coefs_instruments(mreg)
        if regimes:
            summary_regimes(mreg, chow=False)
        summary_warning(mreg)
        multireg[m].__summary = mreg.__summary
    reg.__summary = {}
    if regimes:
        summary_chow(reg, lambd=True)
    summary_warning(reg)
    summary_multi(reg=reg, multireg=multireg, vm=vm,
                  instruments=True, nonspat_diag=False, spat_diag=False)


def Probit(reg, vm, w, spat_diag):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    beta_diag(reg, None)
    # organize summary output
    if spat_diag:
        reg.__summary['summary_spat_diag'] = summary_spat_diag_probit(reg)
    reg.__summary[
        'summary_r2'] = "%-21s: %3.2f\n" % ('% correctly predicted', reg.predpc)
    reg.__summary[
        'summary_r2'] += "%-21s: %3.4f\n" % ('Log-Likelihood', reg.logl)
    reg.__summary['summary_r2'] += "%-21s: %3.4f\n" % ('LR test', reg.LR[0])
    reg.__summary[
        'summary_r2'] += "%-21s: %3.4f\n" % ('LR test (p-value)', reg.LR[1])
    if reg.warning:
        reg.__summary[
            'summary_r2'] += "\nMaximum number of iterations exceeded or gradient and/or function calls not changing\n"
    # build coefficients table body
    beta_position = summary_coefs_allx(reg, reg.z_stat)
    reg.__summary['summary_other_mid'] = summary_coefs_slopes(reg)
    summary(reg=reg, vm=vm, instruments=False,
            short_intro=True, spat_diag=spat_diag)

def SUR(reg, nonspat_diag=True, spat_diag=False, regimes=False,\
        tsls=False, lambd=False, surlm=False):
    reg.__summary = {}
    # compute diagnostics and organize summary output
    reg.__summary['summary_zt'] = 'z'
    reg.__summary['summary_std_err'] = None
    if not tsls:
        try:
            sum_str = "%-20s:%12.3f                %-22s:%12d\n" % (
                'Log likelihood (SUR)', reg.llik, 'Number of Iterations', reg.niter)
        except:
            sum_str = "%-20s:%12.3f\n" % (
                'Log likelihood (SUR)', reg.llik)
        summary_add_other_top(reg, sum_str)
    if lambd:
        sum_str = "%-20s:%12.3f                %-22s:%12.3f\n" % (
            'Log likel. (error)', reg.errllik, 'Log likel. (SUR error)', reg.surerrllik)
        summary_add_other_top(reg, sum_str)
    # build coefficients table body
    summary_coefs_sur(reg, lambd=lambd)
    if tsls:
        for i in range(1,reg.n_eq+1):
            summary_coefs_instruments(reg, sur=i)
    #if regimes:
    #    summary_regimes(reg)
    summary_warning(reg)
    summary_diag_sur(reg, nonspat_diag=nonspat_diag, spat_diag=spat_diag, tsls=tsls, lambd=lambd)
    if surlm and spat_diag:  # only in classic SUR
        sum_str =  summary_spat_diag_intro(no_mi=True)
        sum_str += "%-27s      %2d    %12.3f        %6.4f\n" % (
        "Lagrange Multiplier (error)", reg.lmEtest[1], reg.lmEtest[0], reg.lmEtest[2])
        summary_add_other_end(reg, sum_str)

    summary_errorcorr(reg)
    summary_SUR(reg=reg)


##############################################################################


##############################################################################
############### Helper functions for running summary diagnostics #############
##############################################################################

def beta_diag_ols(reg, robust):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.t_stat = diagnostics.t_stat(reg)
    reg.r2 = diagnostics.r2(reg)
    reg.ar2 = diagnostics.ar2(reg)
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 't'
    reg.__summary['summary_r2'] = "%-20s:%12.4f\n%-20s:%12.4f\n" % (
        'R-squared', reg.r2, 'Adjusted R-squared', reg.ar2)
    # build coefficients table body
    position = summary_coefs_allx(reg, reg.t_stat)


def beta_diag(reg, robust):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.z_stat = diagnostics.t_stat(reg, z_stat=True)
    reg.pr2 = diagnostics_tsls.pr2_aspatial(reg)
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 'z'
    reg.__summary[
        'summary_r2'] = "%-20s:%12.4f\n" % ('Pseudo R-squared', reg.pr2)

def beta_diag_lag(reg, robust, error=True):
    # compute diagnostics
    reg.std_err = diagnostics.se_betas(reg)
    reg.z_stat = diagnostics.t_stat(reg, z_stat=True)
    reg.pr2 = diagnostics_tsls.pr2_aspatial(reg)
    # organize summary output
    reg.__summary['summary_std_err'] = robust
    reg.__summary['summary_zt'] = 'z'
    reg.__summary[
        'summary_r2'] = "%-20s:      %5.4f\n" % ('Pseudo R-squared', reg.pr2)
    if np.abs(reg.rho) < 1:
        reg.pr2_e = diagnostics_tsls.pr2_spatial(reg)
        reg.__summary[
            'summary_r2'] += "%-20s:  %5.4f\n" % ('Spatial Pseudo R-squared', reg.pr2_e)
    else:
        reg.__summary[
            'summary_r2'] += "Spatial Pseudo R-squared: omitted due to rho outside the boundary (-1, 1)."


def build_coefs_body_instruments(reg):
    beta_position = summary_coefs_allx(reg, reg.z_stat)
    summary_coefs_allx(reg, reg.z_stat)
    summary_coefs_instruments(reg)


def spat_diag_ols(reg, w, moran):
    # compute diagnostics
    lm_tests = diagnostics_sp.LMtests(reg, w)
    reg.lm_error = lm_tests.lme
    reg.lm_lag = lm_tests.lml
    reg.rlm_error = lm_tests.rlme
    reg.rlm_lag = lm_tests.rlml
    reg.lm_sarma = lm_tests.sarma
    if moran:
        moran_res = diagnostics_sp.MoranRes(reg, w, z=True)
        reg.moran_res = moran_res.I, moran_res.zI, moran_res.p_norm
    # organize summary output
    reg.__summary['summary_spat_diag'] = summary_spat_diag_ols(reg, moran)


def spat_diag_instruments(reg, w):
    # compute diagnostics
    cache = diagnostics_sp.spDcache(reg, w)
    mi, ak, ak_p = diagnostics_sp.akTest(reg, w, cache)
    reg.ak_test = ak, ak_p
    # organize summary output
    reg.__summary['summary_spat_diag'] = "%-27s      %2d    %12.3f       %9.4f\n" % (
        "Anselin-Kelejian Test", 1, reg.ak_test[0], reg.ak_test[1])


def summary(reg, vm, instruments, short_intro=False, nonspat_diag=False, spat_diag=False, other_end=False):
    summary = summary_open()
    summary += summary_intro(reg, short_intro)
    summary += reg.__summary['summary_r2']
    if nonspat_diag:
        summary += reg.__summary['summary_nonspat_diag_1']
    try:
        summary += reg.__summary['summary_other_top']
    except:
        pass
    summary += summary_coefs_intro(reg)
    summary += reg.__summary['summary_coefs']
    summary += "------------------------------------------------------------------------------------\n"
    if instruments:
        summary += reg.__summary['summary_coefs_instruments']
    try:
        summary += reg.__summary['summary_other_mid']
    except:
        pass
    if nonspat_diag:
        summary += reg.__summary['summary_nonspat_diag_2']
    if spat_diag:
        summary += summary_spat_diag_intro()
        summary += reg.__summary['summary_spat_diag']
    if vm:
        summary += summary_vm(reg, instruments)
    try:
        summary += reg.__summary['summary_chow']
    except:
        pass
    if other_end:
        summary += reg.__summary['summary_other_end']
    summary += summary_close()
    reg.summary = summary


def summary_multi(reg, multireg, vm, instruments, short_intro=False, nonspat_diag=False, spat_diag=False, other_end=False):
    summary = summary_open(multi=True)
    for m in sorted(multireg):
        mreg = multireg[m]
        summary += "----------\n\n"
        summary += summary_intro(mreg, short_intro)
        summary += mreg.__summary['summary_r2']
        if nonspat_diag:
            summary += mreg.__summary['summary_nonspat_diag_1']
        try:
            summary += reg.__summary['summary_other_top']
        except:
            pass
        summary += summary_coefs_intro(mreg)
        summary += mreg.__summary['summary_coefs']
        summary += "------------------------------------------------------------------------------------\n"
        if instruments:
            summary += mreg.__summary['summary_coefs_instruments']
        try:
            summary += mreg.__summary['summary_other_mid']
        except:
            pass
        if m == multireg.keys()[-1]:
            try:
                summary += reg.__summary['summary_other_mid']
            except:
                pass
        if nonspat_diag:
            summary += mreg.__summary['summary_nonspat_diag_2']
        if spat_diag:
            summary += summary_spat_diag_intro()
            summary += mreg.__summary['summary_spat_diag']
        if vm:
            summary += summary_vm(mreg, instruments)
        if other_end:
            summary += mreg.__summary['summary_other_end']
        if m == multireg.keys()[-1]:
            try:
                summary += reg.__summary['summary_chow']
            except:
                pass
            if spat_diag:
                try:
                    spat_diag_str = reg.__summary['summary_spat_diag']
                    summary += summary_spat_diag_intro_global()
                    summary += spat_diag_str
                except:
                    pass
            try:
                summary += reg.__summary['summary_other_end']
            except:
                pass
    summary += summary_close()
    reg.summary = summary

def summary_SUR(reg, short_intro=True):
    summary = summary_open()
    summary += summary_intro(reg, short_intro, sur=True)
    summary += "%-20s:%12d                %-22s:%12d\n" % (
        'Number of Equations', reg.n_eq, 'Number of Observations', reg.n)
    try:
        summary += reg.__summary['summary_other_top']
    except:
        pass
    summary += "----------\n\n"
    for m in reg.name_bigy.keys():
        summary += summary_sur_mid(reg, m)
        #summary += mreg.__summary['summary_r2']
        #if nonspat_diag:
        #    summary += mreg.__summary['summary_nonspat_diag_1']
        summary += summary_coefs_intro(reg)
        summary += reg.__summary[m]['summary_coefs']
        summary += "------------------------------------------------------------------------------------\n"
        try:
            summary += reg.__summary[m]['summary_coefs_instruments']
        except:
            pass
        try:
            summary += reg.__summary['summary_other_mid']
        except:
            pass
        summary += "\n"
    try:
        summary += reg.__summary['summary_other_end']
    except:
        pass
    summary += summary_close()
    reg.summary = summary


def _get_var_indices(reg, lambd=False):
    try:
        var_names = reg.name_z
    except:
        var_names = reg.name_x
    last_v = len(var_names)
    if lambd:
        last_v += -1
    indices = []
    try:
        kf = reg.kf
        if lambd:
            kf += -1
        krex = reg.kr - reg.kryd
        try:
            kfyd = reg.yend.shape[1] - reg.nr * reg.kryd
        except:
            kfyd = 0
        j_con = 0
        if reg.constant_regi == 'many':
            j_con = 1
        for i in range(reg.nr):
            j = i * krex
            jyd = krex * reg.nr + i * reg.kryd + kf - kfyd
            name_reg = var_names[j + j_con:j + krex] + \
                var_names[jyd:jyd + reg.kryd]
            #name_reg.sort()
            if reg.constant_regi == 'many':
                indices += [j] + [var_names.index(ind) for ind in name_reg]
            else:
                indices += [var_names.index(ind) for ind in name_reg]
        if reg.constant_regi == 'one':
            indices += [krex * reg.nr]
        if len(indices) < last_v:
            name_reg = var_names[krex * reg.nr + 1 - j_con:krex * reg.nr + kf -
                                 kfyd] + var_names[reg.kr * reg.nr + kf - kfyd:reg.kr * reg.nr + kf]
            #name_reg.sort()
            indices += [var_names.index(ind) for ind in name_reg]
    except:
        #indices = [0] + (np.argsort(var_names[1:last_v]) + 1).tolist()
        indices = range(len(var_names[1:last_v])+1)
    return var_names, indices



##############################################################################


##############################################################################
############### Guts of the summary printout #################################
##############################################################################

"""
This section contains the pieces needed to put together the summary printout.
"""


def summary_open(multi=False):
    strSummary = ""
    strSummary += "REGRESSION\n"
    if not multi:
        strSummary += "----------\n"
    return strSummary


def summary_intro(reg, short, sur=False):  # extra space d
    title = "SUMMARY OF OUTPUT: " + reg.title + "\n"
    strSummary = title
    strSummary += "-" * (len(title) - 1) + "\n"
    strSummary += "%-20s:%12s\n" % ('Data set', reg.name_ds)
    try:
        strSummary += "%-20s:%12s\n" % ('Weights matrix', reg.name_w)
    except:
        pass
    if not sur:
        strSummary += "%-20s:%12s                %-22s:%12d\n" % (
            'Dependent Variable', reg.name_y, 'Number of Observations', reg.n)
    if not short:
        strSummary += "%-20s:%12.4f                %-22s:%12d\n" % (
            'Mean dependent var', reg.mean_y, 'Number of Variables', reg.k)
        strSummary += "%-20s:%12.4f                %-22s:%12d\n" % (
            'S.D. dependent var', reg.std_y, 'Degrees of Freedom', reg.n - reg.k)
    #strSummary += '\n'
    return strSummary

def summary_sur_mid(reg, eq):
    strSummary = "SUMMARY OF EQUATION " + str(eq+1) + "\n"
    strSummary += "-" * (len(strSummary) - 1) + "\n"
    n_var = int(reg.bigK[reg.name_bigy.keys().index(eq)])
    strSummary += "%-20s:%12s                %-22s:%12d\n" % (
        'Dependent Variable', reg.name_bigy[eq], 'Number of Variables', n_var)
    strSummary += "%-20s:%12.4f                %-22s:%12d\n" % (
        'Mean dependent var', np.mean(reg.bigy[eq]), 'Degrees of Freedom', reg.n - n_var)
    strSummary += "%-20s:%12.4f\n" % (
        'S.D. dependent var', np.std(reg.bigy[eq]))
    return strSummary


def summary_coefs_intro(reg):
    strSummary = "\n"
    if reg.__summary['summary_std_err']:
        if reg.__summary['summary_std_err'].lower() == 'white':
            strSummary += "White Standard Errors\n"
        elif reg.__summary['summary_std_err'].lower() == 'hac':
            strSummary += "HAC Standard Errors; Kernel Weights: " + \
                reg.name_gwk + "\n"
        # elif reg.__summary['summary_std_err'].lower() == 'het':
            #strSummary += "Heteroskedastic Corrected Standard Errors\n"
    strSummary += "------------------------------------------------------------------------------------\n"
    strSummary += "            Variable     Coefficient       Std.Error     %1s-Statistic     Probability\n" % (
        reg.__summary['summary_zt'])
    strSummary += "------------------------------------------------------------------------------------\n"
    return strSummary


def summary_coefs_allx(reg, zt_stat, lambd=False):
    strSummary = ""
    var_names, indices = _get_var_indices(reg, lambd)
    for i in indices:
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
            % (var_names[i], reg.betas[i][0], reg.std_err[i], zt_stat[i][0], zt_stat[i][1])
    reg.__summary['summary_coefs'] = strSummary

    return i

def summary_coefs_sur(reg, lambd=False):
    try:
        betas = reg.bSUR
        inf = reg.sur_inf
    except:
        betas = reg.b3SLS
        inf = reg.tsls_inf
    for eq in reg.name_bigy.keys():
        reg.__summary[eq] = {}
        strSummary = ""
        for i in range(len(reg.name_bigX[eq])):
            strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                % (reg.name_bigX[eq][i], betas[eq][i][0], inf[eq][i][0],\
                     inf[eq][i][1], inf[eq][i][2])
        try:
            i += 1
            for j in range(len(reg.name_bigyend[eq])):
                strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                    % (reg.name_bigyend[eq][j], betas[eq][i+j][0], inf[eq][i+j][0],\
                         inf[eq][i+j][1], inf[eq][i+j][2])
        except:
            pass 
        if lambd:
            pos = reg.name_bigy.keys().index(eq)
            try:
                strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                    % ("lambda_"+str(eq+1), reg.lamsur[pos], reg.lamsetp[0][pos][0],\
                         reg.lamsetp[1][pos][0], reg.lamsetp[2][pos][0])
            except:
                strSummary += "%20s    %12.7f    \n"   \
                    % ("lambda_"+str(eq+1), reg.lamsur[pos])

        reg.__summary[eq]['summary_coefs'] = strSummary

def summary_coefs_somex(reg, zt_stat):
    """This is a special case needed for models that do not have inference on
    the lambda term
    """
    strSummary = ""
    var_names, indices = _get_var_indices(reg, lambd=True)
    for i in indices:
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
            % (reg.name_x[i], reg.betas[i][0], reg.std_err[i], zt_stat[i][0], zt_stat[i][1])
    reg.__summary['summary_coefs'] = strSummary
    return i


'''
def summary_coefs_yend(reg, zt_stat, lambd=False):
    strSummary = ""
    indices = _get_var_indices(reg, lambd) 
    for i in indices:
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
                     % (reg.name_z[i],reg.betas[i][0],reg.std_err[i],zt_stat[i][0],zt_stat[i][1])              
    reg.__summary['summary_coefs'] = strSummary
'''


def summary_coefs_lambda(reg, zt_stat):
    try:
        name_var = reg.name_z
    except:
        name_var = reg.name_x
    if len(reg.betas) == len(zt_stat):
        reg.__summary['summary_coefs'] += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
            % (name_var[-1], reg.betas[-1][0], reg.std_err[-1], zt_stat[-1][0], zt_stat[-1][1])
    else:
        reg.__summary[
            'summary_coefs'] += "%20s    %12.7f    \n" % (name_var[-1], reg.betas[-1][0])


def summary_coefs_instruments(reg, sur=None):
    """Generates a list of the instruments used.
    """
    if not sur:
        name_q = reg.name_q
        name_yend = reg.name_yend
    else:
        eq = reg.name_bigy.keys()[sur-1]
        name_q = reg.name_bigq[eq]
        name_yend = reg.name_bigyend[eq]
    insts = "Instruments: "
    for name in sorted(name_q):
        insts += name + ", "
    text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
    insts = text_wrapper.fill(insts[:-2])
    insts += "\n"
    inst2 = "Instrumented: "
    for name in sorted(name_yend):
        inst2 += name + ", "
    text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
    inst2 = text_wrapper.fill(inst2[:-2])
    inst2 += "\n"
    inst2 += insts
    if not sur:
        reg.__summary['summary_coefs_instruments'] = inst2
    else:
        reg.__summary[eq]['summary_coefs_instruments'] = inst2


def summary_iteration(reg):  # extra space d
    """Reports the number of iterations computed.
    """
    try:
        niter = reg.niter
    except:
        niter = reg.iteration        
    try:
        if reg.step1c:
            step1c = 'Yes'
        else:
            step1c = 'No'
        txt = "%-20s:%12s                %-22s:%12s\n" % (
            'N. of iterations', niter, 'Step1c computed', step1c)
    except:
        txt = "%-20s:%12s\n" % ('N. of iterations', niter)

    summary_add_other_top(reg, txt)


def summary_add_other_top(reg, sum_str):
    try:
        reg.__summary['summary_other_top'] += sum_str
    except:
        reg.__summary['summary_other_top'] = sum_str


def summary_regimes(reg, chow=True):
    """Lists the regimes variable used.
    """
    try:
        reg.__summary[
            'summary_other_mid'] += "Regimes variable: %s\n" % reg.name_regimes
    except:
        reg.__summary[
            'summary_other_mid'] = "Regimes variable: %s\n" % reg.name_regimes
    if chow:
        summary_chow(reg)

''' 
deprecated
def summary_sur(reg, u_cov=False):
    """Lists the equation ID variable used.
    """
    if u_cov:
        str_ucv = "\nERROR COVARIANCE MATRIX\n"
        for i in range(reg.u_cov.shape[0]):
            for j in range(reg.u_cov.shape[1]):
                str_ucv += "%12.6f" % (reg.u_cov[i][j])
            str_ucv += "\n"
        try:
            reg.__summary['summary_other_end'] += str_ucv
        except:
            reg.__summary['summary_other_end'] = str_ucv
    else:
        try:
            reg.__summary[
                'summary_other_mid'] += "Equation ID: %s\n" % reg.name_multiID
        except:
            reg.__summary[
                'summary_other_mid'] = "Equation ID: %s\n" % reg.name_multiID
        try:
            reg.__summary[
                'summary_r2'] += "%-20s: %3.4f\n" % ('Log-Likelihood', reg.logl)
        except:
            pass
'''

def summary_chow(reg, lambd=False):
    reg.__summary['summary_chow'] = "\nREGIMES DIAGNOSTICS - CHOW TEST\n"
    reg.__summary[
        'summary_chow'] += "                 VARIABLE        DF        VALUE           PROB\n"
    if reg.cols2regi == 'all':
        names_chow = reg.name_x_r[1:]
    else:
        names_chow = [reg.name_x_r[1:][i] for i in np.where(reg.cols2regi)[0]]
    if reg.constant_regi == 'many':
        indices = [0] + (np.argsort(names_chow) + 1).tolist()
        names_chow = ['CONSTANT'] + names_chow
    else:
        indices = (np.argsort(names_chow)).tolist()
    if lambd:
        indices += [-1]
        names_chow += ['lambda']
    for i in indices:
        reg.__summary['summary_chow'] += "%25s        %2d    %12.3f        %9.4f\n" % (
            names_chow[i], reg.nr - 1, reg.chow.regi[i, 0], reg.chow.regi[i, 1])
    reg.__summary['summary_chow'] += "%25s        %2d    %12.3f        %9.4f\n" % (
        'Global test', reg.kr * (reg.nr - 1), reg.chow.joint[0], reg.chow.joint[1])


def summary_warning(reg):
    try:
        if reg.warning:
            try:
                reg.__summary['summary_other_mid'] += reg.warning
            except:
                reg.__summary['summary_other_mid'] = reg.warning
    except:
        pass


def summary_coefs_slopes(reg):
    strSummary = "\nMARGINAL EFFECTS\n"
    if reg.scalem == 'phimean':
        strSummary += "Method: Mean of individual marginal effects\n"
    elif reg.scalem == 'xmean':
        strSummary += "Method: Marginal effects at variables mean\n"
    strSummary += "------------------------------------------------------------------------------------\n"
    strSummary += "            Variable           Slope       Std.Error     %1s-Statistic     Probability\n" % (
        reg.__summary['summary_zt'])
    strSummary += "------------------------------------------------------------------------------------\n"
    indices = np.argsort(reg.name_x[1:]).tolist()
    for i in indices:
        strSummary += "%20s    %12.7f    %12.7f    %12.7f    %12.7f\n"   \
            % (reg.name_x[i + 1], reg.slopes[i][0], reg.slopes_std_err[i], reg.slopes_z_stat[i][0], reg.slopes_z_stat[i][1])
    return strSummary + "\n\n"

"""
def summary_r2(reg, ols, spatial_lag):
    if ols:
        strSummary = "%-20s:%12.4f\n%-20s:%12.4f\n" % ('R-squared',reg.r2,'Adjusted R-squared',reg.ar2)
    else:
        strSummary = "%-20s:%12.4f\n" % ('Pseudo R-squared',reg.pr2)
        if spatial_lag:
            if reg.pr2_e != None: 
                strSummary += "%-20s:%12.4f\n" % ('Spatial Pseudo R-squared',reg.pr2_e)
    return strSummary
"""


def summary_nonspat_diag_1(reg):  # extra space d
    strSummary = ""
    strSummary += "%-20s:%12.3f                %-22s:%12.4f\n" % (
        'Sum squared residual', reg.utu, 'F-statistic', reg.f_stat[0])
    strSummary += "%-20s:%12.3f                %-22s:%12.4g\n" % (
        'Sigma-square', reg.sig2, 'Prob(F-statistic)', reg.f_stat[1])
    strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'S.E. of regression', np.sqrt(reg.sig2), 'Log likelihood', reg.logll)
    strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
        'Sigma-square ML', reg.sig2ML, 'Akaike info criterion', reg.aic)
    strSummary += "%-20s:%12.4f                %-22s:%12.3f\n" % (
        'S.E of regression ML', np.sqrt(reg.sig2ML), 'Schwarz criterion', reg.schwarz)
    return strSummary


def summary_nonspat_diag_2(reg):
    strSummary = ""
    strSummary += "\nREGRESSION DIAGNOSTICS\n"
    if reg.mulColli:
        strSummary += "MULTICOLLINEARITY CONDITION NUMBER %16.3f\n\n" % (
            reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d  %14.3f        %9.4f\n\n" % (
        'Jarque-Bera', reg.jarque_bera['df'], reg.jarque_bera['jb'], reg.jarque_bera['pvalue'])
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        'Breusch-Pagan test', reg.breusch_pagan['df'], reg.breusch_pagan['bp'], reg.breusch_pagan['pvalue'])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        'Koenker-Bassett test', reg.koenker_bassett['df'], reg.koenker_bassett['kb'], reg.koenker_bassett['pvalue'])
    try:
        if reg.white:
            strSummary += "\nSPECIFICATION ROBUST TEST\n"
            if len(reg.white) > 3:
                strSummary += reg.white + '\n'
            else:
                strSummary += "TEST                             DF        VALUE           PROB\n"
                strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
                    'White', reg.white['df'], reg.white['wh'], reg.white['pvalue'])
    except:
        pass
    return strSummary


def summary_spat_diag_intro(no_mi=False):
    strSummary = ""
    strSummary += "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
    if no_mi:
        strSummary += "TEST                              DF       VALUE           PROB\n"
    else:
        strSummary += "TEST                           MI/DF       VALUE           PROB\n"
    return strSummary


def summary_spat_diag_intro_global():
    strSummary = ""
    strSummary += "\nDIAGNOSTICS FOR GLOBAL SPATIAL DEPENDENCE\n"
    strSummary += "Residuals are treated as homoskedastic for the purpose of these tests\n"
    strSummary += "TEST                           MI/DF       VALUE           PROB\n"
    return strSummary


def summary_spat_diag_ols(reg, moran):
    strSummary = ""
    if moran:
        strSummary += "%-27s  %8.4f     %9.3f        %9.4f\n" % (
            "Moran's I (error)", reg.moran_res[0], reg.moran_res[1], reg.moran_res[2])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        "Lagrange Multiplier (lag)", 1, reg.lm_lag[0], reg.lm_lag[1])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        "Robust LM (lag)", 1, reg.rlm_lag[0], reg.rlm_lag[1])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        "Lagrange Multiplier (error)", 1, reg.lm_error[0], reg.lm_error[1])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n" % (
        "Robust LM (error)", 1, reg.rlm_error[0], reg.rlm_error[1])
    strSummary += "%-27s      %2d    %12.3f        %9.4f\n\n" % (
        "Lagrange Multiplier (SARMA)", 2, reg.lm_sarma[0], reg.lm_sarma[1])
    return strSummary


def summary_spat_diag_probit(reg):
    strSummary = ""
    strSummary += "%-27s      %2d    %12.3f       %9.4f\n" % (
        "Kelejian-Prucha (error)", 1, reg.KP_error[0], reg.KP_error[1])
    strSummary += "%-27s      %2d    %12.3f       %9.4f\n" % (
        "Pinkse (error)", 1, reg.Pinkse_error[0], reg.Pinkse_error[1])
    strSummary += "%-27s      %2d    %12.3f       %9.4f\n\n" % (
        "Pinkse-Slade (error)", 1, reg.PS_error[0], reg.PS_error[1])
    return strSummary

def summary_diag_sur(reg, nonspat_diag, spat_diag, tsls, lambd, rho=False):
    if tsls:
        try:
            if reg.joinrho != None:
                rho = True
        except:
            pass
        if spat_diag and rho:
            strSummary = ""
            strSummary += "\nREGRESSION DIAGNOSTICS\n"
            #strSummary += "----------------------\n"
            strSummary += "                                     TEST         DF       VALUE           PROB\n"
            strSummary += "%41s        %2d   %10.3f           %6.4f\n" % (
                "Joint significance (rho)", reg.joinrho[1], reg.joinrho[0], reg.joinrho[2])        
            summary_add_other_end(reg, strSummary)
    else:
        if nonspat_diag or (spat_diag and lambd):
            strSummary = ""
            strSummary += "\nREGRESSION DIAGNOSTICS\n"
            #strSummary += "----------------------\n"
            strSummary += "                                     TEST         DF       VALUE           PROB\n"
            if nonspat_diag:
                try:
                    strSummary += "%41s        %2d   %10.3f           %6.4f\n" % (
                        "LM test on Sigma", reg.lmtest[1], reg.lmtest[0], reg.lmtest[2])
                except:
                    pass
                strSummary += "%41s        %2d   %10.3f           %6.4f\n" % (
                    "LR test on Sigma", reg.lrtest[1], reg.lrtest[0], reg.lrtest[2])
            if lambd and spat_diag:
                strSummary += "%41s        %2d   %10.3f           %6.4f\n" % (
                    "LR test on lambda", reg.likrlambda[1], reg.likrlambda[0], reg.likrlambda[2])
                if reg.vm != None:
                    strSummary += "%41s        %2d   %10.3f           %6.4f\n" % (
                        "Joint significance (lambda)", reg.joinlam[1], reg.joinlam[0], reg.joinlam[2])        
            summary_add_other_end(reg, strSummary)
    chow_lamb = False
    try:
        if reg.lamtest != None:
            chow_lamb = True
    except:
        pass
    if nonspat_diag or (spat_diag and chow_lamb):
        if reg.surchow != None:
            strChow = "\nOTHER DIAGNOSTICS - CHOW TEST\n"
            strChow += "                                VARIABLES         DF       VALUE           PROB\n"
            if nonspat_diag:
                kx = len(reg.surchow)
                if tsls:
                    kx += -len(reg.name_bigyend[reg.name_bigyend.keys()[0]])
                names_chow = {}
                for k in range(len(reg.surchow)):
                    names_chow[k] = ""
                    if k < kx:
                        for i in reg.name_bigX:
                            names_chow[k] += reg.name_bigX[i][k] + ", "
                    else:
                        for j in reg.name_bigyend:
                            names_chow[k] += reg.name_bigyend[j][k-kx] + ", "
                    if len(names_chow[k])>42:
                        names_chow[k] = names_chow[k][0:38] + "..."
                    else:
                        names_chow[k] = names_chow[k][:-2]
                for k in range(len(reg.surchow)):
                    strChow += "%41s        %2d   %10.3f           %6.4f\n" % (
                        names_chow[k], reg.surchow[k][1], reg.surchow[k][0], reg.surchow[k][2])
            if spat_diag and chow_lamb:
                strChow += "%41s        %2d   %10.3f           %6.4f\n" % (
                        "lambda", reg.lamtest[1], reg.lamtest[0], reg.lamtest[2])
            summary_add_other_end(reg, strChow)

def summary_add_other_end(reg, strSummary):
    try:
        reg.__summary['summary_other_end'] += strSummary
    except:
        reg.__summary['summary_other_end'] = strSummary

def summary_vm(reg, instruments):
    strVM = "\n"
    strVM += "COEFFICIENTS VARIANCE MATRIX\n"
    strVM += "----------------------------\n"
    if instruments:
        for name in reg.name_z:
            strVM += "%12s" % (name)
    else:
        for name in reg.name_x:
            strVM += "%12s" % (name)
    strVM += "\n"
    nrow = reg.vm.shape[0]
    ncol = reg.vm.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            strVM += "%12.6f" % (reg.vm[i][j])
        strVM += "\n"
    return strVM

def summary_errorcorr(reg):
    str_ucv = "\n"
    str_ucv += "ERROR CORRELATION MATRIX\n"
    #str_ucv += "------------------------\n"
    for i in range(reg.n_eq):
        str_ucv += "%12s" %("EQUATION " + str(i+1))
    str_ucv += "\n"
    nrow = reg.corr.shape[0]
    ncol = reg.corr.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            str_ucv += "%12.6f" % (reg.corr[i][j])
        str_ucv += "\n"
    summary_add_other_end(reg, str_ucv)

def summary_pred(reg):
    strPred = "\n\n"
    strPred += "%16s%16s%16s%16s\n" % ('OBS',
                                       reg.name_y, 'PREDICTED', 'RESIDUAL')
    for i in range(reg.n):
        strPred += "%16d%16.5f%16.5f%16.5f\n" % (
            i + 1, reg.y[i][0], reg.predy[i][0], reg.u[i][0])
    return strPred


def summary_close():
    return "================================ END OF REPORT ====================================="

##############################################################################


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
