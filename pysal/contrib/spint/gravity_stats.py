# coding=utf-8
"""
Statistics for gravity models

References
----------

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations
 and Applications. London: Kluwer Academic Publishers.

Williams, P. A. and A. S. Fotheringham (1984), The Calibration of Spatial Interaction
 Models by Maximum Likelihood Estimation with Program SIMODEL, Geographic Monograph
 Series, 7, Department of Geography, Indiana University.

Wilson, A. G. (1967). A statistical theory of spatial distribution models.
 Transportation Research, 1, 253–269.


"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import gravity as gv


def sys_stats(gm):
    """
    calculate descriptive statistics of model system
    """
    system_stats = {}
    num_origins = len(gm.o.unique())
    system_stats['num_origins'] = num_origins
    num_destinations = len(gm.d.unique())
    system_stats['num_destinations'] = num_destinations
    pairs = len(gm.dt)
    system_stats['OD_pairs'] = pairs
    observed_flows = np.sum(gm.f)
    system_stats['observed_flows'] = observed_flows
    predicted_flows = np.sum(gm.ests)
    system_stats['predicted_flows'] = predicted_flows
    avg_dist = round(np.sum(gm.c)*float(1)/pairs)
    system_stats['avg_dist'] = avg_dist
    avg_dist_trav = round((np.sum(gm.f*gm.c))*float(1)/np.sum(gm.f)*float(1))
    system_stats['avg_dist_trav'] = avg_dist_trav
    obs_mean_trip_len = (np.sum(gm.f*gm.c))*float(1)/observed_flows*float(1)
    system_stats['obs_mean_trip_len'] = obs_mean_trip_len
    pred_mean_trip_len = (np.sum(gm.ests*gm.c))/predicted_flows
    system_stats['pred_mean_trip_len'] = pred_mean_trip_len
    return system_stats

def ent_stats(gm):
    """
    calculate the entropy statistics for the model system
    """
    entropy_stats = {}
    pij = gm.f/np.sum(gm.f)
    phatij = gm.ests/np.sum(gm.f)
    max_ent = round(np.log(len(gm.dt)), 4)
    entropy_stats['maximum_entropy'] = max_ent
    pred_ent = round(-np.sum(phatij*np.log(phatij)), 4)
    entropy_stats['predicted_entropy'] = pred_ent
    obs_ent = round(-np.sum(pij*np.log(pij)), 4)
    entropy_stats['observed_entropy'] = obs_ent
    diff_pred_ent = round(max_ent - pred_ent, 4)
    entropy_stats['max_pred_deviance'] = diff_pred_ent
    diff_obs_ent = round(max_ent - obs_ent, 4)
    entropy_stats['max_obs_deviance'] = diff_obs_ent
    diff_ent = round(pred_ent - obs_ent, 4)
    entropy_stats['pred_obs_deviance'] = diff_ent
    ent_rs = round(diff_pred_ent/diff_obs_ent, 4)
    entropy_stats['entropy_ratio'] = ent_rs
    obs_flows = np.sum(gm.f)
    var_pred_ent = round(((np.sum(phatij*(np.log(phatij)**2))-pred_ent**2)/obs_flows) + ((len(gm.dt)-1)/(2*obs_flows**2)), 11)
    entropy_stats['variance_pred_entropy'] = var_pred_ent
    var_obs_ent = round(((np.sum(pij*np.log(pij)**2)-obs_ent**2)/obs_flows) + ((len(gm.dt)-1)/(2*obs_flows**2)), 11)
    entropy_stats['variance_obs_entropy'] = var_obs_ent
    t_stat_ent = round((pred_ent-obs_ent)/((var_pred_ent+var_obs_ent)**.5), 4)
    entropy_stats['t_stat_entropy'] = t_stat_ent
    return entropy_stats

def fit_stats(gm):
    """
    calculate the goodness-of-fit statistics
    """
    fit_stats = {}
    srmse = ((np.sum((gm.f-gm.ests)**2)/len(gm.dt))**.5)/(np.sum(gm.f)/len(gm.dt))
    fit_stats['srmse'] = srmse
    pearson_r = pearsonr(gm.ests, gm.f)[0]
    fit_stats['r_squared'] = pearson_r**2
    return fit_stats


def param_stats(gm):
    """
    calculate standard errors and likelihood statistics
    """
    parameter_statistics = {}
    PV = list(gm.p.values())
    if len(PV) == 1:
        first_deriv = gv.o_function(PV, gm, gm.cf, gm.of, gm.df)
        recalc_fd = gv.o_function([PV[0]+.001], gm, gm.cf, gm.of, gm.df)
        diff = first_deriv[0]-recalc_fd[0]
        second_deriv = -(1/(diff/.001))
        gm.p['beta'] = PV
        parameter_statistics['beta'] = {}
        parameter_statistics['beta']['standard_error'] = np.sqrt(second_deriv)
    elif len(PV) > 1:
        var_matrix = np.zeros((len(PV),len(PV)))
        for x, param in enumerate(PV):
            first_deriv = gv.o_function(PV, gm, gm.cf, gm.of, gm.df)
            var_params = list(PV)
            var_params[x] += .001
            var_matrix[x] = gv.o_function(var_params, gm, gm.cf, gm.of, gm.df)
            var_matrix[x] = (first_deriv-var_matrix[x])/.001
        errors = np.sqrt(-np.linalg.inv(var_matrix).diagonal())
        for x, param in enumerate(gm.p):
            parameter_statistics[param] = {'standard_error': errors[x]}

    LL = np.sum((gm.f/np.sum(gm.f))*np.log((gm.ests/np.sum(gm.ests))))
    parameter_statistics['all_params'] = {}
    parameter_statistics['all_params']['mle_vals_LL'] = LL
    new_PV = list(PV)
    for x, param in enumerate(gm.p):
        new_PV[x] = 0
        gv.o_function(new_PV, gm, gm.cf, gm.of, gm.df)
        LL_ests = gm.estimate_flows(gm.c, gm.cf, gm.of, gm.df, gm.p)
        new_LL = np.sum((gm.f/np.sum(gm.f))*np.log((LL_ests/np.sum(LL_ests))))
        parameter_statistics[param]['LL_zero_val'] = new_LL
        lamb = 2*np.sum(gm.f)*(LL-new_LL)
        parameter_statistics[param]['relative_likelihood_stat'] = lamb
        new_PV = list(PV)
    for x, param in enumerate(PV):
        new_PV[x] = 0
    gv.o_function(new_PV, gm, gm.cf, gm.of, gm.df)
    LL_ests = gm.estimate_flows(gm.c, gm.cf, gm.of, gm.df, gm.p)
    LL_zero = np.sum((gm.f/np.sum(gm.f))*np.log((LL_ests/np.sum(LL_ests))))
    parameter_statistics['all_params']['zero_vals_LL'] = LL_zero
    return parameter_statistics



