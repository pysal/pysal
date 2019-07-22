"""
Local based Segregation Metrics

Important: all classes that start with "Multi_" expects a specific type of input of multigroups since the index will be calculated using many groups.
On the other hand, other classes expects a single group for calculation of the metrics.
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pysal.lib as lps

from pysal.explore.segregation.spatial import RelativeCentralization

from pysal.explore.segregation.util.util import _dep_message, DeprecationHelper

# Including old and new api in __all__ so users can use both

__all__ = [
    'Multi_Location_Quotient',
    'MultiLocationQuotient',
    
    'Multi_Local_Diversity',
    'MultiLocalDiversity',
    
    'Multi_Local_Entropy',
    'MultiLocalEntropy',
    
    'Multi_Local_Simpson_Interaction',
    'MultiLocalSimpsonInteraction',
    
    'Multi_Local_Simpson_Concentration',
    'MultiLocalSimpsonConcentration',
    
    'Local_Relative_Centralization',
    'LocalRelativeCentralization'
]

# The Deprecation calls of the classes are located in the end of this script #




# suppress numpy divide by zero warnings because it occurs a lot during the
# calculation of many indices
np.seterr(divide='ignore', invalid='ignore')

def _multi_location_quotient(data, groups):
    """
    Calculation of Location Quotient index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n,k)
                 Location Quotient values for each group and unit.
                 Column k has the Location Quotient of position k in groups.
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Isard, Walter. Methods of regional analysis. Vol. 4. Рипол Классик, 1967.
    
    Reference: :cite:`isard1967methods`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    n = df.shape[0]
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    Xk = df.sum(axis = 0)
    
    multi_LQ = (df / np.repeat(ti, K, axis = 0).reshape(n,K)) / (Xk / T)
    
    return multi_LQ, core_data


class MultiLocationQuotient:
    """
    Calculation of Location Quotient index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n,k)
                 Location Quotient values for each group and unit.
                 Column k has the Location Quotient of position k in groups.
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.
                 
    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import MultiLocationQuotient
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = MultiLocationQuotient(input_df, groups_list)
    >>> index.statistics[0:3,0:3]
    array([[1.36543221, 0.07478049, 0.16245651],
           [1.18002164, 0.        , 0.14836683],
           [0.68072696, 0.03534425, 0.        ]])

    Important to note that column k has the Location Quotient (LQ) of position k in groups. Therefore, the LQ of the first unit of 'WHITE_' is 1.36543221.
    
    Notes
    -----
    Based on Isard, Walter. Methods of regional analysis. Vol. 4. Рипол Классик, 1967.
    
    Reference: :cite:`isard1967methods`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_location_quotient(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_location_quotient
        
        
        

def _multi_local_diversity(data, groups):
    """
    Calculation of Local Diversity index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n,k)
                 Local Diversity values for each group and unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Theil, Henry. Statistical decomposition analysis; with applications in the social and administrative sciences. No. 04; HA33, T4.. 1972.
    
    Reference: :cite:`theil1972statistical`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    multi_LD = - np.nansum(pik * np.log(pik), axis = 1)
    
    return multi_LD, core_data


class MultiLocalDiversity:
    """
    Calculation of Local Diversity index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n,k)
                 Local Diversity values for each group and unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.
                 
    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import MultiLocalDiversity
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = MultiLocalDiversity(input_df, groups_list)
    >>> index.statistics[0:10] # Values of first 10 units
    array([0.34332326, 0.56109229, 0.70563225, 0.29713472, 0.22386084,
           0.29742517, 0.12322789, 0.11274579, 0.09402405, 0.25129616])

    Notes
    -----
    Based on Theil, Henry. Statistical decomposition analysis; with applications in the social and administrative sciences. No. 04; HA33, T4.. 1972.
    
    Reference: :cite:`theil1972statistical`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_diversity(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_diversity
        
        
def _multi_local_entropy(data, groups):
    """
    Calculation of Local Entropy index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Entropy values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Eq. 6 of pg. 139 (individual unit case) of Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    Reference: :cite:`reardon2004measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    K = df.shape[1]
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    multi_LE = - np.nansum((pik * np.log(pik)) / np.log(K), axis = 1)
    
    return multi_LE, core_data


class MultiLocalEntropy:
    """
    Calculation of Local Entropy index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Entropy values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import MultiLocalEntropy
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = MultiLocalEntropy(input_df, groups_list)
    >>> index.statistics[0:10] # Values of first 10 units
    array([0.24765538, 0.40474253, 0.50900607, 0.21433739, 0.16148146,
           0.21454691, 0.08889013, 0.08132889, 0.06782401, 0.18127186])

    Notes
    -----
    Based on Eq. 6 of pg. 139 (individual unit case) of Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    Reference: :cite:`reardon2004measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_entropy(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_entropy
        


def _multi_local_simpson_interaction(data, groups):
    """
    Calculation of Local Simpson Interaction index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Simpson Interaction values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    local_SI = np.nansum(pik * (1 - pik), axis = 1)
    
    return local_SI, core_data


class MultiLocalSimpsonInteraction:
    """
    Calculation of Local Simpson Interaction index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Simpson Interaction values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import MultiLocalSimpsonInteraction
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = MultiLocalSimpsonInteraction(input_df, groups_list)
    >>> index.statistics[0:10] # Values of first 10 units
    array([0.15435993, 0.33391595, 0.49909747, 0.1299449 , 0.09805056,
           0.13128178, 0.04447356, 0.0398933 , 0.03723054, 0.11758548])

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_simpson_interaction(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_simpson_interaction
        
        
def _multi_local_simpson_concentration(data, groups):
    """
    Calculation of Local Simpson concentration index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Simpson concentration values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's concentration index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    local_SC = np.nansum(pik * pik, axis = 1)
    
    return local_SC, core_data


class MultiLocalSimpsonConcentration:
    """
    Calculation of Local Simpson concentration index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Simpson concentration values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import MultiLocalSimpsonConcentration
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = MultiLocalSimpsonConcentration(input_df, groups_list)
    >>> index.statistics[0:10] # Values of first 10 units
    array([0.84564007, 0.66608405, 0.50090253, 0.8700551 , 0.90194944,
           0.86871822, 0.95552644, 0.9601067 , 0.96276946, 0.88241452])

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's concentration index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_simpson_concentration(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_simpson_concentration
        
        
def _local_relative_centralization(data, group_pop_var, total_pop_var, k_neigh = 5):
    """
    Calculation of Local Relative Centralization index for each unit

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    k_neigh       : integer greater than 0. Default is 5.
                    Number of assumed neighbors for local context (it uses k-nearest algorithm method)
                    
    Returns
    -------

    statistics : np.array(n)
                 Local Relative Centralization values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """
    
    data = data.copy()
    
    core_data = data[[group_pop_var, total_pop_var, 'geometry']]

    c_lons = data.centroid.map(lambda p: p.x)
    c_lats = data.centroid.map(lambda p: p.y)
    
    points = list(zip(c_lons, c_lats))
    kd = lps.cg.kdtree.KDTree(np.array(points))
    wnnk = lps.weights.KNN(kd, k = k_neigh)
    
    local_RCEs = np.empty(len(data))
    
    for i in range(len(data)):
    
        x = list(wnnk.neighbors.values())[i]
        x.append(list(wnnk.neighbors.keys())[i]) # Append in the end the current unit i inside the local context

        local_data = data.iloc[x,:].copy()
        
        # The center is given by the last position (i.e. the current unit i)
        local_RCE = RelativeCentralization(local_data, group_pop_var, total_pop_var, center = len(local_data) - 1)
        
        local_RCEs[i] = local_RCE.statistic
        
    return local_RCEs, core_data


class LocalRelativeCentralization:
    """
    Calculation of Local Relative Centralization index for each unit

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    k_neigh       : integer greater than 0. Default is 5.
                    Number of assumed neighbors for local context (it uses k-nearest algorithm method)
                    
    Returns
    -------

    statistics : np.array(n)
                 Local Relative Centralization values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.
                 
    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The group of interest is Black population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pysal.lib
    >>> import geopandas as gpd
    >>> from pysal.explore.segregation.local import LocalRelativeCentralization
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
    
    The value is estimated below.
    
    >>> index = LocalRelativeCentralization(input_df, 'BLACK_', 'TOT_POP')
    >>> index.statistics[0:10] # Values of first 10 units
    array([ 0.03443055, -0.29063264, -0.19110976,  0.24978919,  0.01252249,
            0.61152941,  0.78917647,  0.53129412,  0.04436346, -0.20216325])

    Notes
    -----
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """
    
    def __init__(self, data, group_pop_var, total_pop_var, k_neigh = 5):
        
        aux = _local_relative_centralization(data, group_pop_var, total_pop_var, k_neigh)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _local_relative_centralization
        
        
        




# Deprecation Calls

msg = _dep_message("Multi_Location_Quotient", "MultiLocationQuotient")
Multi_Location_Quotient = DeprecationHelper(MultiLocationQuotient, message=msg)

msg = _dep_message("Multi_Local_Diversity", "MultiLocalDiversity")
Multi_Local_Diversity = DeprecationHelper(MultiLocalDiversity, message=msg)

msg = _dep_message("Multi_Local_Entropy", "MultiLocalEntropy")
Multi_Local_Entropy = DeprecationHelper(MultiLocalEntropy, message=msg)

msg = _dep_message("Multi_Local_Simpson_Interaction", "MultiLocalSimpsonInteraction")
Multi_Local_Simpson_Interaction = DeprecationHelper(MultiLocalSimpsonInteraction, message=msg)

msg = _dep_message("Multi_Local_Simpson_Concentration", "MultiLocalSimpsonConcentration")
Multi_Local_Simpson_Concentration = DeprecationHelper(MultiLocalSimpsonConcentration, message=msg)

msg = _dep_message("Local_Relative_Centralization", "LocalRelativeCentralization")
Local_Relative_Centralization = DeprecationHelper(LocalRelativeCentralization, message=msg)