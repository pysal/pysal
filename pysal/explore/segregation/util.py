"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com> and Renan X. Cortes <renanc@ucr.edu>"

import numpy as np
import pandas as pd
import pysal.lib
import geopandas as gpd
from warnings import warn
from pysal.lib.weights import Queen, KNN
from pysal.lib.weights.util import attach_islands


def _return_length_weighted_w(data):
    """
    Returns a PySAL weights object that the weights represent the length of the commom boudary of two areal units that share border.

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.

    Notes
    -----
    Currently it's not making any projection.

    """
    
    w = pysal.lib.weights.Rook.from_dataframe(data, 
                                             ids = data.index.tolist(),
                                             geom_col = data._geometry_column_name)
    
    if (len(w.islands) == 0):
        w = w
    else:
        warn('There are some islands in the GeoDataFrame.')
        w_aux = pysal.lib.weights.KNN.from_dataframe(data, 
                                                    ids = data.index.tolist(),
                                                    geom_col = data._geometry_column_name,
                                                    k = 1)
        w = attach_islands(w, w_aux)
    
    adjlist = w.to_adjlist()
    islands = pd.DataFrame.from_records([{'focal':island, 'neighbor':island, 'weight':0} for island in w.islands])
    merged = adjlist.merge(data.geometry.to_frame('geometry'), left_on='focal',
                           right_index=True, how='left')\
                    .merge(data.geometry.to_frame('geometry'), left_on='neighbor',
                           right_index=True, how='left', suffixes=("_focal", "_neighbor"))\
    
    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry='geometry_focal')
    merged['geometry_neighbor'] = gpd.GeoSeries(merged.geometry_neighbor)
    
    # Getting the shared boundaries
    merged['shared_boundary'] = merged.geometry_focal.intersection(merged.set_geometry('geometry_neighbor'))
    
    # Putting it back to a matrix
    merged['weight'] = merged.set_geometry('shared_boundary').length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = pysal.lib.weights.W.from_adjlist(merged_with_islands[['focal', 'neighbor', 'weight']])
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]
    
    length_weighted_w._reset()
    
    return length_weighted_w



def _generate_counterfactual(data1, data2, group_pop_var, total_pop_var, counterfactual_approach = 'composition'):
    
    """Generate a counterfactual variables.

    Given two contexts, generate counterfactual distributions for a variable of
    interest by simulating the variable of one context into the spatial
    structure of the other.

    Parameters
    ----------
    data1 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 1
        
    data2 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 2
        
    group_pop_var : str
        The name of variable in both data that contains the population size of the group of interest
        
    total_pop_var : str
        The name of variable in both data that contains the total population of the unit
        
    approach : str, ["composition", "share", "dual_composition"]
        Which approach to use for generating the counterfactual.
        Options include "composition", "share", or "dual_composition"

    Returns
    -------
    two DataFrames
        df1 and df2  with appended columns 'counterfactual_group_pop', 'counterfactual_total_pop', 'group_composition' and 'counterfactual_composition'

    """
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
        
    if ((group_pop_var not in data1.columns) or (total_pop_var not in data1.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data1')
        
    if ((group_pop_var not in data2.columns) or (total_pop_var not in data2.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data2')
        
    if any(data1[total_pop_var] < data1[group_pop_var]):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units in data1.')
        
    if any(data2[total_pop_var] < data2[group_pop_var]):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units in data2.')
    
    df1 = data1.copy()
    df2 = data2.copy()
    
    if (counterfactual_approach == 'composition'):
    
        df1['group_composition'] = np.where(df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[total_pop_var])
    
        df1['counterfactual_group_pop'] = df1['group_composition'].rank(pct = True).apply(df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(pct = True).apply(df1['group_composition'].quantile) * df2[total_pop_var]
        
        df1['counterfactual_total_pop'] = df1[total_pop_var]
        df2['counterfactual_total_pop'] = df2[total_pop_var]
        
    if (counterfactual_approach == 'share'):
        
        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]
        
        df1['share'] = np.where(df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[group_pop_var].sum())
        df2['share'] = np.where(df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[group_pop_var].sum())

        df1['compl_share'] = np.where(df1['compl_pop_var'] == 0, 0, df1['compl_pop_var'] / df1['compl_pop_var'].sum())
        df2['compl_share'] = np.where(df2['compl_pop_var'] == 0, 0, df2['compl_pop_var'] / df2['compl_pop_var'].sum())
        
        df1['counterfactual_group_pop'] = df1['share'].rank(pct = True).apply(df2['share'].quantile) * df1[group_pop_var].sum()
        df2['counterfactual_group_pop'] = df2['share'].rank(pct = True).apply(df1['share'].quantile) * df2[group_pop_var].sum()
        
        df1['counterfactual_compl_pop'] = df1['compl_share'].rank(pct = True).apply(df2['compl_share'].quantile) * df1['compl_pop_var'].sum()
        df2['counterfactual_compl_pop'] = df2['compl_share'].rank(pct = True).apply(df1['compl_share'].quantile) * df2['compl_pop_var'].sum()
        
        df1['counterfactual_total_pop'] = df1['counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2['counterfactual_group_pop'] + df2['counterfactual_compl_pop']
        
    if (counterfactual_approach == 'dual_composition'):
        
        df1['group_composition'] = np.where(df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[total_pop_var])
        
        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]
        
        df1['compl_composition'] = np.where(df1[total_pop_var] == 0, 0, df1['compl_pop_var'] / df1[total_pop_var])
        df2['compl_composition'] = np.where(df2[total_pop_var] == 0, 0, df2['compl_pop_var'] / df2[total_pop_var])
        
        df1['counterfactual_group_pop'] = df1['group_composition'].rank(pct = True).apply(df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(pct = True).apply(df1['group_composition'].quantile) * df2[total_pop_var]
        
        df1['counterfactual_compl_pop'] = df1['compl_composition'].rank(pct = True).apply(df2['compl_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_compl_pop'] = df2['compl_composition'].rank(pct = True).apply(df1['compl_composition'].quantile) * df2[total_pop_var]
        
        df1['counterfactual_total_pop'] = df1['counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2['counterfactual_group_pop'] + df2['counterfactual_compl_pop']
    
    df1['group_composition'] = np.where(df1['total_pop_var'] == 0, 0, df1['group_pop_var'] / df1['total_pop_var'])
    df2['group_composition'] = np.where(df2['total_pop_var'] == 0, 0, df2['group_pop_var'] / df2['total_pop_var'])
    
    df1['counterfactual_composition'] = np.where(df1['counterfactual_total_pop'] == 0, 0, df1['counterfactual_group_pop'] / df1['counterfactual_total_pop'])
    df2['counterfactual_composition'] = np.where(df2['counterfactual_total_pop'] == 0, 0, df2['counterfactual_group_pop'] / df2['counterfactual_total_pop'])
    
    df1 = df1.drop(columns=['group_pop_var', 'total_pop_var'], axis = 1)
    df2 = df2.drop(columns=['group_pop_var', 'total_pop_var'], axis = 1)
    
    return df1, df2