"""
Inference Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"


import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

__all__ = ['Infer_Segregation',
           'Compare_Segregation']

def _infer_segregation(seg_class, iterations_under_null = 500, null_approach = "systematic", two_tailed = True, **kwargs):
    '''
    Perform inference for a single segregation measure

    Parameters
    ----------

    seg_class                    : a PySAL segregation object
    
    iterations_under_null        : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hypothesis the inference will iterate.
    
        "systematic"             : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
        "evenness"               : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
        
        "permutation"            : randomly allocates the units over space keeping the original values.
        
        "systematic_permutation" : assumes absence of systematic segregation and randomly allocates the units over space.
        "even_permutation"       : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    
    two_tailed    : boolean
                    If True, p_value is two-tailed. Otherwise, it is right one-tailed.
    
    **kwargs      : customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.
    
    Attributes
    ----------

    p_value     : float
                  Pseudo One or Two-Tailed p-value estimated from the simulations
    
    est_sim     : numpy array
                  Estimates of the segregation measure under the null hypothesis
                  
    statistic   : float
                  The point estimation of the segregation measure that is under test
                
    Notes
    -----
    The one-tailed p_value attribute might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the est_sim attribute.
    
    '''
    if not null_approach in ['systematic', 'evenness', 'permutation', 'systematic_permutation', 'even_permutation']:
        raise ValueError('null_approach must one of \'systematic\', \'evenness\', \'permutation\', \'systematic_permutation\', \'even_permutation\'')
    
    if (type(two_tailed) is not bool):
        raise TypeError('two_tailed is not a boolean object')
    
    point_estimation = seg_class.statistic
    data             = seg_class.core_data
    
    aux = str(type(seg_class))
    _class_name = aux[1 + aux.rfind('.'):-2] # 'rfind' finds the last occurence of a pattern in a string

    
    ##############
    # SYSTEMATIC #
    ##############
    if (null_approach == "systematic"):
    
        data['other_group_pop'] = data['total_pop_var'] - data['group_pop_var']
        p_j = data['total_pop_var'] / data['total_pop_var'].sum()

        # Group 0: minority group
        p0_i = p_j
        n0 = data['group_pop_var'].sum()
        sim0 = np.random.multinomial(n0, p0_i, size = iterations_under_null)

        # Group 1: complement group
        p1_i = p_j
        n1 = data['other_group_pop'].sum()
        sim1 = np.random.multinomial(n1, p1_i, size = iterations_under_null)

        Estimates_Stars = np.empty(iterations_under_null)
        
        for i in np.array(range(iterations_under_null)):
            data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            
            if (str(type(data)) == '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
                
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")
    
    ############
    # EVENNESS #
    ############
    if (null_approach == "evenness"):
        
        p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()
        
        Estimates_Stars = np.empty(iterations_under_null)
        
        for i in np.array(range(iterations_under_null)):
            sim = np.random.binomial(n = np.array([data['total_pop_var'].tolist()]), 
                                     p = p_null)
            data_aux = {'simul_group': sim[0], 'simul_tot': data['total_pop_var'].tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            
            if (str(type(data)) == '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
            
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")
        
    ###############
    # PERMUTATION #
    ###############
    if (null_approach == "permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
        
        Estimates_Stars = np.empty(iterations_under_null)
        
        for i in np.array(range(iterations_under_null)):
            data = data.assign(geometry = data['geometry'][list(np.random.choice(data.shape[0], data.shape[0], replace = False))].reset_index()['geometry'])
            df_aux = data
            Estimates_Stars[i] = seg_class._function(df_aux, 'group_pop_var', 'total_pop_var', **kwargs)[0]
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")

    ##########################
    # SYSTEMATIC PERMUTATION #
    ##########################
    if (null_approach == "systematic_permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
    
        data['other_group_pop'] = data['total_pop_var'] - data['group_pop_var']
        p_j = data['total_pop_var'] / data['total_pop_var'].sum()

        # Group 0: minority group
        p0_i = p_j
        n0 = data['group_pop_var'].sum()
        sim0 = np.random.multinomial(n0, p0_i, size = iterations_under_null)

        # Group 1: complement group
        p1_i = p_j
        n1 = data['other_group_pop'].sum()
        sim1 = np.random.multinomial(n1, p1_i, size = iterations_under_null)

        Estimates_Stars = np.empty(iterations_under_null)
        
        for i in np.array(range(iterations_under_null)):
            data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            df_aux = gpd.GeoDataFrame(df_aux)
            df_aux['geometry'] = data['geometry']
            df_aux = df_aux.assign(geometry = df_aux['geometry'][list(np.random.choice(df_aux.shape[0], df_aux.shape[0], replace = False))].reset_index()['geometry'])
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]  
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")
    
    ########################
    # EVENNESS PERMUTATION #
    ########################
    if (null_approach == "even_permutation"):
        
        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')
        
        p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()
        
        Estimates_Stars = np.empty(iterations_under_null)
        
        for i in np.array(range(iterations_under_null)):
            sim = np.random.binomial(n = np.array([data['total_pop_var'].tolist()]), 
                                     p = p_null)
            data_aux = {'simul_group': sim[0], 'simul_tot': data['total_pop_var'].tolist()}
            df_aux = pd.DataFrame.from_dict(data_aux)
            df_aux = gpd.GeoDataFrame(df_aux)
            df_aux['geometry'] = data['geometry']
            df_aux = df_aux.assign(geometry = df_aux['geometry'][list(np.random.choice(df_aux.shape[0], df_aux.shape[0], replace = False))].reset_index()['geometry'])
            Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")

    
    # Check and, if the case, remove iterations_under_null that resulted in nan or infinite values
    if any((np.isinf(Estimates_Stars) | np.isnan(Estimates_Stars))):
        warnings.warn('Some estimates resulted in NaN or infinite values for estimations under null hypothesis. These values will be removed for the final results.')
        Estimates_Stars = Estimates_Stars[~(np.isinf(Estimates_Stars) | np.isnan(Estimates_Stars))]
    
    if not two_tailed:
        p_value = sum(Estimates_Stars > point_estimation) / iterations_under_null
    else:
        aux1 = (point_estimation < Estimates_Stars).sum()
        aux2 = (point_estimation > Estimates_Stars).sum()
        p_value = 2 * np.array([aux1, aux2]).min() / len(Estimates_Stars)
    
    return p_value, Estimates_Stars, point_estimation, _class_name



class Infer_Segregation:
    '''
    Perform inference for a single segregation measure

    Parameters
    ----------

    seg_class                    : a PySAL segregation object
    
    iterations_under_null        : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hypothesis the inference will iterate.
    
        "systematic"             : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
        "evenness"               : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
        
        "permutation"            : randomly allocates the units over space keeping the original values.
        
        "systematic_permutation" : assumes absence of systematic segregation and randomly allocates the units over space.
        "even_permutation"       : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    
    two_tailed    : boolean
                    If True, p_value is two-tailed. Otherwise, it is right one-tailed.
    
    **kwargs      : customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.
    
    Attributes
    ----------

    p_value     : float
                  Pseudo One or Two-Tailed p-value estimated from the simulations
    
    est_sim     : numpy array
                  Estimates of the segregation measure under the null hypothesis
                  
    statistic   : float
                  The point estimation of the segregation measure that is under test
                
    Notes
    -----
    The one-tailed p_value attribute might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the est_sim attribute.
    
    Examples
    --------
    Several examples can be found here https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb.
    
    '''

    def __init__(self, seg_class, iterations_under_null = 500, null_approach = "systematic", two_tailed = True, **kwargs):
        
        aux = _infer_segregation(seg_class, iterations_under_null, null_approach, two_tailed, **kwargs)

        self.p_value      = aux[0]
        self.est_sim      = aux[1]
        self.statistic    = aux[2]
        self._class_name  = aux[3]
        
    def plot(self):
        """
        Plot the Infer_Segregation class
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib` and `seaborn`')
    
        sns.distplot(self.est_sim, 
                     hist = True, 
                     color = 'darkblue', 
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 2})
        plt.axvline(self.statistic, color = 'red')
        plt.title('{} (Value = {})'.format(self._class_name, round(self.statistic, 3)))
        return plt.show()
    
    
    
    
    
    
    
    
def _compare_segregation(seg_class_1, seg_class_2, iterations_under_null = 500, null_approach = "random_label", **kwargs):
    '''
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1           : a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2           : a PySAL segregation object to be compared to seg_class_1
    
    iterations_under_null : number of iterations under null hyphothesis
    
    null_approach: argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_label"               : random label the data in each iteration
        
        "counterfactual_composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

        "counterfactual_share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.

    **kwargs : customizable parameters to pass to the segregation measures. Usually they need to be the same as both seg_class_1 and seg_class_2  was built.
    
    Attributes
    ----------

    p_value        : float
                     Two-Tailed p-value
    
    est_sim        : numpy array
                     Estimates of the segregation measure differences under the null hypothesis
                  
    est_point_diff : float
                     Point estimation of the difference between the segregation measures
                
    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either two measures of the same locations in two different points in time or it can be two different locations at the same point in time.
    
    The null hypothesis is H0: Segregation_1 is not different than Segregation_2.
    
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.

    '''
    
    if not null_approach in ['random_label', 'counterfactual_composition', 'counterfactual_share']:
        raise ValueError('null_approach must one of \'random_label\', \'counterfactual_composition\', \'counterfactual_share\'')
    
    if(type(seg_class_1) != type(seg_class_2)):
        raise TypeError('seg_class_1 and seg_class_2 must be the same type/class.')
    
    point_estimation = seg_class_1.statistic - seg_class_2.statistic
    
    aux = str(type(seg_class_1))
    _class_name = aux[1 + aux.rfind('.'):-2]  # 'rfind' finds the last occurence of a pattern in a string

    data_1 = seg_class_1.core_data
    data_2 = seg_class_2.core_data
    
    # This step is just to make sure the each frequecy column is integer for the approaches and from the same type in order to stack them for the random data approach
    data_1['group_pop_var'] = round(data_1['group_pop_var']).astype(int)
    data_1['total_pop_var'] = round(data_1['total_pop_var']).astype(int)
    
    data_2['group_pop_var'] = round(data_2['group_pop_var']).astype(int)
    data_2['total_pop_var'] = round(data_2['total_pop_var']).astype(int)
    
    est_sim = np.empty(iterations_under_null)
    
    
    
    ################
    # RANDOM LABEL #
    ################
    if (null_approach == "random_label"):

        data_1['grouping_variable'] = 'Group_1'
        data_2['grouping_variable'] = 'Group_2'

        stacked_data = pd.concat([data_1, data_2], ignore_index=True)
        
        for i in np.array(range(iterations_under_null)):
            
            aux_rand = list(np.random.choice(stacked_data.shape[0], stacked_data.shape[0], replace = False))

            stacked_data['rand_group_pop'] = stacked_data.group_pop_var[aux_rand].reset_index()['group_pop_var']
            stacked_data['rand_total_pop'] = stacked_data.total_pop_var[aux_rand].reset_index()['total_pop_var']
            
            # Dropping variable to avoid confusion in the calculate_segregation function 
            # Building auxiliar data to avoid affecting the next iteration
            stacked_data_aux = stacked_data.drop(['group_pop_var', 'total_pop_var'], axis = 1)
            
            stacked_data_1 = stacked_data_aux.loc[stacked_data_aux['grouping_variable'] == 'Group_1']
            stacked_data_2 = stacked_data_aux.loc[stacked_data_aux['grouping_variable'] == 'Group_2']

            simulations_1 = seg_class_1._function(stacked_data_1, 'rand_group_pop', 'rand_total_pop', **kwargs)[0]
            simulations_2 = seg_class_2._function(stacked_data_2, 'rand_group_pop', 'rand_total_pop', **kwargs)[0]
            
            est_sim[i] = simulations_1 - simulations_2
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")
            

            
            
    ##############################
    # COUNTERFACTUAL COMPOSITION #
    ##############################
    if (null_approach == "counterfactual_composition"):

        data_1['rel'] = np.where(data_1['total_pop_var'] == 0, 0, data_1['group_pop_var'] / data_1['total_pop_var'])
        data_2['rel'] = np.where(data_2['total_pop_var'] == 0, 0, data_2['group_pop_var'] / data_2['total_pop_var'])

        # Both appends are to force both distribution to have values in all space between 0 and 1
        x_1_pre = np.sort(data_1['rel'])
        y_1_pre = np.arange(0, len(x_1_pre)) / (len(x_1_pre))

        x_2_pre = np.sort(data_2['rel'])
        y_2_pre = np.arange(0, len(x_2_pre)) / (len(x_2_pre))

        x_1 = np.append(np.append(0, x_1_pre), 1)
        y_1 = np.append(np.append(0, y_1_pre), 1)

        x_2 = np.append(np.append(0, x_2_pre), 1)
        y_2 = np.append(np.append(0, y_2_pre), 1)

        def inverse_cdf_1(pct):
            return x_1[np.where(y_1 > pct)[0][0] - 1]

        def inverse_cdf_2(pct):
            return x_2[np.where(y_2 > pct)[0][0] - 1]

        # Adding the pseudo columns for FIRST spatial context
        data_1['cumulative_percentage'] = (data_1['rel'].rank() - 1) / len(data_1) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_1['pseudo_rel'] = data_1['cumulative_percentage'].apply(inverse_cdf_2)
        data_1['pseudo_group_pop_var'] = round(data_1['pseudo_rel'] * data_1['total_pop_var']).astype(int)

        # Adding the pseudo columns for SECOND spatial context
        data_2['cumulative_percentage'] = (data_2['rel'].rank() - 1) / len(data_2) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_2['pseudo_rel'] = data_2['cumulative_percentage'].apply(inverse_cdf_1)
        data_2['pseudo_group_pop_var'] = round(data_2['pseudo_rel'] * data_2['total_pop_var']).astype(int)

        for i in np.array(range(iterations_under_null)):

            data_1['fair_coin'] = np.random.uniform(size = len(data_1))
            data_1['test_group_pop_var'] = np.where(data_1['fair_coin'] > 0.5, data_1['group_pop_var'], data_1['pseudo_group_pop_var'])
            
            # Dropping to avoid confusion in the internal function
            data_1_test = data_1.drop(['group_pop_var'], axis = 1)
            
            
            simulations_1 = seg_class_1._function(data_1_test, 'test_group_pop_var', 'total_pop_var', **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_1 = data_1.drop(['fair_coin', 'test_group_pop_var'], axis = 1)
            

            
            data_2['fair_coin'] = np.random.uniform(size = len(data_2))
            data_2['test_group_pop_var'] = np.where(data_2['fair_coin'] > 0.5, data_2['group_pop_var'], data_2['pseudo_group_pop_var'])
            
            # Dropping to avoid confusion in the internal function
            data_2_test = data_2.drop(['group_pop_var'], axis = 1)
            
            simulations_2 = seg_class_2._function(data_2_test, 'test_group_pop_var', 'total_pop_var', **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_2 = data_2.drop(['fair_coin', 'test_group_pop_var'], axis = 1)
            
            
            est_sim[i] = simulations_1 - simulations_2
            
            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r") 
            
            
            
            
            
            
            
    ########################
    # COUNTERFACTUAL SHARE #
    ########################
    if (null_approach == "counterfactual_share"):

        data_1['compl_pop_var'] = data_1['total_pop_var'] - data_1['group_pop_var']
        data_2['compl_pop_var'] = data_2['total_pop_var'] - data_2['group_pop_var']
        
        # Build the share for each group individually
        data_1['share'] = np.where(data_1['total_pop_var'] == 0, 0, data_1['group_pop_var'] / data_1['group_pop_var'].sum())
        data_2['share'] = np.where(data_2['total_pop_var'] == 0, 0, data_2['group_pop_var'] / data_2['group_pop_var'].sum())

        data_1['compl_share'] = np.where(data_1['compl_pop_var'] == 0, 0, data_1['compl_pop_var'] / data_1['compl_pop_var'].sum())
        data_2['compl_share'] = np.where(data_2['compl_pop_var'] == 0, 0, data_2['compl_pop_var'] / data_2['compl_pop_var'].sum())

        # Both appends are to force both distribution to have values in all space between 0 and 1
        x_1_pre = np.sort(data_1['share'])
        y_1_pre = np.arange(0, len(x_1_pre)) / (len(x_1_pre))

        x_2_pre = np.sort(data_2['share'])
        y_2_pre = np.arange(0, len(x_2_pre)) / (len(x_2_pre))

        x_1 = np.append(np.append(0, x_1_pre), 1)
        y_1 = np.append(np.append(0, y_1_pre), 1)

        x_2 = np.append(np.append(0, x_2_pre), 1)
        y_2 = np.append(np.append(0, y_2_pre), 1)

        def inverse_cdf_1(pct):
            return x_1[np.where(y_1 > pct)[0][0] - 1]

        def inverse_cdf_2(pct):
            return x_2[np.where(y_2 > pct)[0][0] - 1]



        # Both appends are to force both distribution to have values in all space between 0 and 1
        compl_x_1_pre = np.sort(data_1['compl_share'])
        compl_y_1_pre = np.arange(0, len(compl_x_1_pre)) / (len(compl_x_1_pre))

        compl_x_2_pre = np.sort(data_2['compl_share'])
        compl_y_2_pre = np.arange(0, len(compl_x_2_pre)) / (len(compl_x_2_pre))

        compl_x_1 = np.append(np.append(0, compl_x_1_pre), 1)
        compl_y_1 = np.append(np.append(0, compl_y_1_pre), 1)

        compl_x_2 = np.append(np.append(0, compl_x_2_pre), 1)
        compl_y_2 = np.append(np.append(0, compl_y_2_pre), 1)

        def compl_inverse_cdf_1(pct):
            return compl_x_1[np.where(compl_y_1 > pct)[0][0] - 1]

        def compl_inverse_cdf_2(pct):
            return compl_x_2[np.where(compl_y_2 > pct)[0][0] - 1]


        # Adding the pseudo columns for FIRST spatial context
        data_1['cumulative_percentage'] = (data_1['share'].rank() - 1) / len(data_1) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_1['pseudo_share_pre']      = data_1['cumulative_percentage'].apply(inverse_cdf_2)
        data_1['pseudo_share']          = data_1['pseudo_share_pre'] / data_1['pseudo_share_pre'].sum() # Rescale due to possibility of the summation of the values being grater of lower than 1
        data_1['pseudo_group_pop_var']  = round(data_1['pseudo_share'] * data_1['group_pop_var'].sum()).astype(int)


        data_1['compl_cumulative_percentage'] = (data_1['compl_share'].rank() - 1) / len(data_1) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_1['compl_pseudo_share_pre']      = data_1['compl_cumulative_percentage'].apply(compl_inverse_cdf_2)
        data_1['compl_pseudo_share']          = data_1['compl_pseudo_share_pre'] / data_1['compl_pseudo_share_pre'].sum() # Rescale due to possibility of the summation of the values being grater of lower than 1
        data_1['pseudo_compl_pop_var']        = round(data_1['compl_pseudo_share'] * data_1['compl_pop_var'].sum()).astype(int)


        data_1['pseudo_total_pop'] = data_1['pseudo_group_pop_var'] + data_1['pseudo_compl_pop_var']




        # Adding the pseudo columns for SECOND spatial context
        data_2['cumulative_percentage'] = (data_2['share'].rank() - 1) / len(data_2) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_2['pseudo_share_pre']      = data_2['cumulative_percentage'].apply(inverse_cdf_1)
        data_2['pseudo_share']          = data_2['pseudo_share_pre'] / data_2['pseudo_share_pre'].sum() # Rescale due to possibility of the summation of the values being grater of lower than 1
        data_2['pseudo_group_pop_var']  = round(data_2['pseudo_share'] * data_2['group_pop_var'].sum()).astype(int)


        data_2['compl_cumulative_percentage'] = (data_2['compl_share'].rank() - 1) / len(data_2) # It has to be a minus 1 in the rank, in order to avoid 100% percentile in the max
        data_2['compl_pseudo_share_pre']      = data_2['compl_cumulative_percentage'].apply(compl_inverse_cdf_1)
        data_2['compl_pseudo_share']          = data_2['compl_pseudo_share_pre'] / data_2['compl_pseudo_share_pre'].sum() # Rescale due to possibility of the summation of the values being grater of lower than 1
        data_2['pseudo_compl_pop_var']        = round(data_2['compl_pseudo_share'] * data_2['compl_pop_var'].sum()).astype(int)


        data_2['pseudo_total_pop'] = data_2['pseudo_group_pop_var'] + data_2['pseudo_compl_pop_var']

        for i in np.array(range(iterations_under_null)):

            # For this 'counterfactual_share' approach, also the group and total population can be swapped during the iterations
            data_1['fair_coin'] = np.random.uniform(size = len(data_1))
            data_1['test_group_pop_var'] = np.where(data_1['fair_coin'] > 0.5, data_1['group_pop_var'], data_1['pseudo_group_pop_var'])
            data_1['test_total_pop_var'] = np.where(data_1['fair_coin'] > 0.5, data_1['total_pop_var'], data_1['pseudo_total_pop'])
            
            # Dropping to avoid confusion in the internal function
            data_1_test = data_1.drop(['group_pop_var', 'total_pop_var'], axis = 1)
            
            
            simulations_1 = seg_class_1._function(data_1_test, 'test_group_pop_var', 'test_total_pop_var', **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_1 = data_1.drop(['fair_coin', 'test_group_pop_var', 'test_total_pop_var'], axis = 1)
            

            # For this 'counterfactual_share' approach, also the group and total population can be swapped during the iterations
            data_2['fair_coin'] = np.random.uniform(size = len(data_2))
            data_2['test_group_pop_var'] = np.where(data_2['fair_coin'] > 0.5, data_2['group_pop_var'], data_2['pseudo_group_pop_var'])
            data_2['test_total_pop_var'] = np.where(data_2['fair_coin'] > 0.5, data_2['total_pop_var'], data_2['pseudo_total_pop'])
            
            
            # Dropping to avoid confusion in the internal function
            data_2_test = data_2.drop(['group_pop_var', 'total_pop_var'], axis = 1)
            
            simulations_2 = seg_class_2._function(data_2_test, 'test_group_pop_var', 'test_total_pop_var', **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_2 = data_2.drop(['fair_coin', 'test_group_pop_var', 'test_total_pop_var'], axis = 1)
            
            
            est_sim[i] = simulations_1 - simulations_2

            print('Processed {} iterations out of {}.'.format(i + 1, iterations_under_null), end = "\r")            
            
            
            
            
    # Check and, if the case, remove iterations_under_null that resulted in nan or infinite values
    if any((np.isinf(est_sim) | np.isnan(est_sim))):
        warnings.warn('Some estimates resulted in NaN or infinite values for estimations under null hypothesis. These values will be removed for the final results.')
        est_sim = est_sim[~(np.isinf(est_sim) | np.isnan(est_sim))]

    # Two-Tailed p-value
    # Obs.: the null distribution can be located far from zero. Therefore, this is the the appropriate way to calculate the two tailed p-value.
    aux1 = (point_estimation < est_sim).sum()
    aux2 = (point_estimation > est_sim).sum()
    p_value = 2 * np.array([aux1, aux2]).min() / len(est_sim)
    
    return p_value, est_sim, point_estimation, _class_name



class Compare_Segregation:
    '''
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1           : a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2           : a PySAL segregation object to be compared to seg_class_1
    
    iterations_under_null : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_label"      : random label the data in each iteration
        
        "counterfactual_composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

        "counterfactual_share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.

    **kwargs : customizable parameters to pass to the segregation measures. Usually they need to be the same as both seg_class_1 and seg_class_2  was built.
    
    Attributes
    ----------

    p_value        : float
                     Two-Tailed p-value
    
    est_sim        : numpy array
                     Estimates of the segregation measure differences under the null hypothesis
                  
    est_point_diff : float
                     Point estimation of the difference between the segregation measures
                
    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either two measures of the same locations in two different points in time or it can be two different locations at the same point in time.
    
    The null hypothesis is H0: Segregation_1 is not different than Segregation_2.
    
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.
    
    Examples
    --------
    Several examples can be found here https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb.

    '''

    def __init__(self, seg_class_1, seg_class_2, iterations_under_null = 500, null_approach = "random_label", **kwargs):
        
        aux = _compare_segregation(seg_class_1, seg_class_2, iterations_under_null, null_approach, **kwargs)

        self.p_value        = aux[0]
        self.est_sim        = aux[1]
        self.est_point_diff = aux[2]
        self._class_name    = aux[3]
        
    def plot(self):
        """
        Plot the Compare_Segregation class
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib` and `seaborn`')
    
        sns.distplot(self.est_sim, 
                     hist = True, 
                     color = 'darkblue', 
                     hist_kws={'edgecolor':'black'},
                     kde_kws={'linewidth': 2})
        plt.axvline(self.est_point_diff, color = 'red')
        plt.title('{} (Diff. value = {})'.format(self._class_name, round(self.est_point_diff, 3)))
        return plt.show()