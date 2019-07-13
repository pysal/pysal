"""
Spatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
import warnings
import pysal.lib
import math

from pysal.lib.weights import Queen
from numpy import inf
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from scipy.ndimage.interpolation import shift

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix

from pysal.explore.segregation.util import _return_length_weighted_w
from pysal.explore.segregation.non_spatial_indexes import _dissim


__all__ = ['Spatial_Prox_Prof',
           'Spatial_Dissim',
           'Boundary_Spatial_Dissim',
           'Perimeter_Area_Ratio_Spatial_Dissim',
           'Spatial_Isolation',
           'Spatial_Exposure',
           'Spatial_Proximity',
           'Relative_Clustering',
           'Delta',
           'Absolute_Concentration',
           'Relative_Concentration',
           'Absolute_Centralization',
           'Relative_Centralization',
           'Spatial_Information_Theory']


def _spatial_prox_profile(data, group_pop_var, total_pop_var, m = 1000):
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Attributes
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if(type(m) is not int):
        raise TypeError('m must be a string.')
        
    if(m < 2):
        raise ValueError('m must be greater than 1.')
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')

    # Create the shortest distance path between two pair of units using Shimbel matrix. This step was well discussed in https://github.com/pysal/segregation/issues/5.    
    w_libpysal = Queen.from_dataframe(data)
    graph = csr_matrix(w_libpysal.full()[0])
    delta = floyd_warshall(csgraph = graph, directed = False)
    
    def calculate_etat(t):
        g_t_i = np.where(data.group_pop_var / data.total_pop_var >= t, True, False)
        k = g_t_i.sum()
        sub_delta_ij = delta[g_t_i,:][:,g_t_i] # i and j only varies in the units subset within the threshold in eta_t of Hong (2014).
        den = sub_delta_ij.sum()
        eta_t = (k**2 - k) / den
        return eta_t
    
    grid = np.linspace(0, 1, m)
    aux = np.array(list(map(calculate_etat, grid)))
    aux[aux == inf] = 0
    aux[aux == -inf] = 0
    curve = np.nan_to_num(aux, 0)
    
    threshold = data.group_pop_var.sum() / data.total_pop_var.sum()
    SPP = ((threshold - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum()/ m)) / (1 - threshold))
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SPP, grid, curve, core_data


class Spatial_Prox_Prof:
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Attributes
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial proximity profile (SPP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    >>> spat_prox_index = Spatial_Prox_Prof(gdf, 'nhblk10', 'pop10')
    >>> spat_prox_index.statistic
    0.11217269612149207
    
    You can plot the profile curve with the plot method.
    
    >>> spat_prox_index.plot()
        
    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m = 1000):
        
        aux = _spatial_prox_profile(data, group_pop_var, total_pop_var, m)

        self.statistic = aux[0]
        self.grid      = aux[1]
        self.curve     = aux[2]
        self.core_data = aux[3]
        self._function = _spatial_prox_profile

    def plot(self):
        """
        Plot the Spatial Proximity Profile
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib`')
        graph = plt.scatter(self.grid, self.curve, s = 0.1)
        return graph
    
    

def _spatial_dissim(data, group_pop_var, total_pop_var, w = None, standardize = False):
    """
    Calculation of Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    w             : W
                    A PySAL weights object. If not provided, Queen contiguity matrix is used.
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default with row standardization.
        

    Attributes
    ----------

    statistic : float
                Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')
        
    if w is None:    
        w_object = Queen.from_dataframe(data)
    else:
        w_object = w
    
    if (not issubclass(type(w_object), pysal.lib.weights.W)):
        raise TypeError('w is not a PySAL weights object')
    
    D = _dissim(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)
    
    if not standardize:
        cij = w_object.full()[0]
    else:
        cij = w_object.full()[0]
        cij = cij / cij.sum(axis = 1).reshape((cij.shape[0], 1))

    # Inspired in (second solution): https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
    # Distance Matrix
    abs_dist = abs(pi[..., np.newaxis] - pi)
        
    # manhattan_distances used to compute absolute distances
    num = np.multiply(abs_dist, cij).sum()
    den = cij.sum()
    SD = D - num / den
    SD
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SD, core_data


class Spatial_Dissim:
    """
    Calculation of Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    w             : W
                    A PySAL weights object. If not provided, Queen contiguity matrix is used.
    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default with row standardization.

    Attributes
    ----------

    statistic : float
                Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.   
                
    Examples
    --------
    In this example, we will calculate the degree of spatial dissimilarity (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset. The neighborhood contiguity matrix is used.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> spatial_dissim_index.statistic
    0.2864885055405311
        
    To use different neighborhood matrices:
        
    >>> from pysal.lib.weights import Rook, KNN
    
    Assuming K-nearest neighbors with k = 4
    
    >>> knn = KNN.from_dataframe(gdf, k=4)
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10', w = knn)
    >>> spatial_dissim_index.statistic
    0.28544347200877285
    
    Assuming Rook contiguity neighborhood
    
    >>> roo = Rook.from_dataframe(gdf)
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10', w = roo)
    >>> spatial_dissim_index.statistic
    0.2866269198707091
            
    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, w = None, standardize = False):
        
        aux = _spatial_dissim(data, group_pop_var, total_pop_var, w, standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_dissim
        
        
        
        
def _boundary_spatial_dissim(data, group_pop_var, total_pop_var, standardize = False):
    """
    Calculation of Boundary Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default without row standardization. That is, directly with border length.
        

    Attributes
    ----------

    statistic : float
                Boundary Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.

    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')
    
    D = _dissim(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(pi = np.where(data.total_pop_var == 0, 0, data.group_pop_var/data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum(axis = 1).reshape((cij.shape[0], 1))

    # manhattan_distances used to compute absolute distances
    num = np.multiply(manhattan_distances(data[['pi']]), cij).sum()
    den = cij.sum()
    BSD = D - num / den
    BSD
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return BSD, core_data


class Boundary_Spatial_Dissim:
    """
    Calculation of Boundary Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default without row standardization. That is, directly with border length.
        

    Attributes
    ----------

    statistic : float
                Boundary Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
         
    Examples
    --------
    In this example, we will calculate the degree of boundary spatial dissimilarity (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> boundary_spatial_dissim_index = Boundary_Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> boundary_spatial_dissim_index.statistic
    0.28869903953453163
            
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize = False):
        
        aux = _boundary_spatial_dissim(data, group_pop_var, total_pop_var, standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _boundary_spatial_dissim
        
        
        
def _perimeter_area_ratio_spatial_dissim(data, group_pop_var, total_pop_var, standardize = True):
    """
    Calculation of Perimeter/Area Ratio Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for standardisation of the weights matrices. 
                    If True, the values of cij in the formulas gets standardized and the overall sum is 1.

    Attributes
    ----------

    statistic : float
                Perimeter/Area Ratio Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.

    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')
    
    D = _dissim(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(pi = np.where(data.total_pop_var == 0, 0, data.group_pop_var/data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum()
   
    peri = data.length
    ai   = data.area
    
    aux_sum = np.add(np.array(list((peri / ai))), np.array(list((peri / ai))).reshape((len(list((peri / ai))),1)))
    
    max_pa = max(peri / ai)
    
    num = np.multiply(np.multiply(manhattan_distances(data[['pi']]), cij), aux_sum).sum()
    den = 4 * max_pa
    
    PARD = D - (num / den)
    PARD
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return PARD, core_data


class Perimeter_Area_Ratio_Spatial_Dissim:
    """
    Calculation of Perimeter/Area Ratio Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for standardisation of the weights matrices. 
                    If True, the values of cij in the formulas gets standardized and the overall sum is 1.
        
    Attributes
    ----------

    statistic : float
                Perimeter/Area Ratio Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.      
                
    Examples
    --------
    In this example, we will calculate the degree of perimeter/area ratio spatial dissimilarity (PARD) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> perimeter_area_ratio_spatial_dissim_index = Perimeter_Area_Ratio_Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> perimeter_area_ratio_spatial_dissim_index.statistic
    0.31260876347432687
            
    Notes
    -----
    Based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize = True):
        
        aux = _perimeter_area_ratio_spatial_dissim(data, group_pop_var, total_pop_var, standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _perimeter_area_ratio_spatial_dissim
        
        
        
def _spatial_isolation(data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
    """
    Calculation of Spatial Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    This measure is also called the distance decay isolation. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')
    
    if (beta < 0):
        raise ValueError('beta must be greater than zero.')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)
    
    X = x.sum()
    
    dist = euclidean_distances(pd.DataFrame({'c_lons': c_lons, 'c_lats': c_lats}))
    np.fill_diagonal(dist, val = (alpha * data.area) ** (beta))
    c = np.exp(-dist)
    
    Pij  = np.multiply(c, t) / np.sum(np.multiply(c, t), axis = 1)
    SxPx = (np.array(x / X) * np.nansum(np.multiply(Pij, np.array(x / t)), axis = 1)).sum()
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SxPx, core_data


class Spatial_Isolation:
    """
    Calculation of Spatial Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial isolation index (SxPx) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_isolation_index = Spatial_Isolation(gdf, 'nhblk10', 'pop10')
    >>> spatial_isolation_index.statistic
    0.07214112078134231
            
    Notes
    -----
    This measure is also called the distance decay isolation. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
        
        aux = _spatial_isolation(data, group_pop_var, total_pop_var, alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_isolation
        
        

def _spatial_exposure(data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
    """
    Calculation of Spatial Exposure index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    This measure is also called the distance decay exposure. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')
    
    if (beta < 0):
        raise ValueError('beta must be greater than zero.')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    y = t - x
    
    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)
    
    X = x.sum()
    
    dist = euclidean_distances(pd.DataFrame({'c_lons': c_lons, 'c_lats': c_lats}))
    np.fill_diagonal(dist, val = (alpha * data.area) ** (beta))
    c = np.exp(-dist)
    
    Pij  = np.multiply(c, t) / np.sum(np.multiply(c, t), axis = 1)
    SxPy = (x / X * np.nansum(np.multiply(Pij, y / t), axis = 1)).sum()
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SxPy, core_data


class Spatial_Exposure:
    """
    Calculation of Spatial Exposure index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial exposure index (SxPy) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_exposure_index = Spatial_Exposure(gdf, 'nhblk10', 'pop10')
    >>> spatial_exposure_index.statistic
    0.9605053172501217
            
    Notes
    -----
    This measure is also called the distance decay exposure. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
        
        aux = _spatial_exposure(data, group_pop_var, total_pop_var, alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_exposure
        
        
def _spatial_proximity(data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
    """
    Calculation of Spatial Proximity index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
    Attributes
    ----------
    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')
    
    if (beta < 0):
        raise ValueError('beta must be greater than zero.')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    T = data.total_pop_var.sum()
    
    data = data.assign(xi = data.group_pop_var,
                       yi = data.total_pop_var - data.group_pop_var,
                       ti = data.total_pop_var,
                       c_lons = data.centroid.map(lambda p: p.x),
                       c_lats = data.centroid.map(lambda p: p.y))
    
    X = data.xi.sum()
    Y = data.yi.sum()
    
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (alpha * data.area) ** (beta))
    c = np.exp(-dist)
    
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    Ptt = ((np.array(data.ti) * c).T * np.array(data.ti)).sum() / T**2
    SP = (X * Pxx + Y * Pyy) / (T * Ptt)
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]   
    
    return SP, core_data


class Spatial_Proximity:
    """
    Calculation of Spatial Proximity index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
    Attributes
    ----------
    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the degree of spatial proximity (SP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_prox_index = Spatial_Proximity(gdf, 'nhblk10', 'pop10')
    >>> spatial_prox_index.statistic
    1.002191883006537
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
        
        aux = _spatial_proximity(data, group_pop_var, total_pop_var, alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_proximity
        
        
def _relative_clustering(data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
    """
    Calculation of Relative Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
    Attributes
    ----------
    statistic : float
                Relative Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')
    
    if (beta < 0):
        raise ValueError('beta must be greater than zero.')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    data = data.assign(xi = data.group_pop_var,
                       yi = data.total_pop_var - data.group_pop_var,
                       c_lons = data.centroid.map(lambda p: p.x),
                       c_lats = data.centroid.map(lambda p: p.y))
    
    X = data.xi.sum()
    Y = data.yi.sum()
    
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (alpha * data.area) ** (beta))
    c = np.exp(-dist)
    
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    RCL = Pxx / Pyy - 1
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return RCL, core_data


class Relative_Clustering:
    """
    Calculation of Relative Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
    Attributes
    ----------
    statistic : float
                Relative Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the relative clustering measure (RCL) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_clust_index = Relative_Clustering(gdf, 'nhblk10', 'pop10')
    >>> relative_clust_index.statistic
    0.12418089857347714
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
        
        aux = _relative_clustering(data, group_pop_var, total_pop_var, alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_clustering
        
        
def _delta(data, group_pop_var, total_pop_var):
    """
    Calculation of Delta index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Delta Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    area = np.array(data.area)
    
    X = x.sum()
    A = area.sum()
    
    DEL = 1/2 * abs(x / X - area / A).sum()
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return DEL, core_data


class Delta:
    """
    Calculation of Delta index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Delta Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the delta index (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> delta_index = Delta(gdf, 'nhblk10', 'pop10')
    >>> delta_index.statistic
    0.8367330649317353
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _delta(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _delta
        
        
def _absolute_concentration(data, group_pop_var, total_pop_var):
    """
    Calculation of Absolute Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Absolute Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    area = np.array(data.area)
    
    X = x.sum()
    T = t.sum()
    
    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()
    
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X/T) == False)[0][0]
    n2 = np.where(((np.cumsum(t[des_ind]) / T) < X/T) == False)[0][0]
    
    n = data.shape[0]
    T1 =  t[asc_ind][0:(n1+1)].sum()
    T2 =  t[asc_ind][n2:n].sum()
    
    ACO = 1- ((((x[asc_ind] * area[asc_ind] / X).sum()) - ((t[asc_ind] * area[asc_ind] / T1)[0:(n1 + 1)].sum())) / \
          (((t[asc_ind] * area[asc_ind] / T2)[n2:n].sum()) - ((t[asc_ind] * area[asc_ind]/T1)[0:(n1 + 1)].sum())))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return ACO, core_data


class Absolute_Concentration:
    """
    Calculation of Absolute Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Absolute Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Examples
    --------
    In this example, we will calculate the absolute concentration index (ACO) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_concentration_index = Absolute_Concentration(gdf, 'nhblk10', 'pop10')
    >>> absolute_concentration_index.statistic
    0.5430616390401855
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _absolute_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_concentration
        
        
def _relative_concentration(data, group_pop_var, total_pop_var):
    """
    Calculation of Relative Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    area = np.array(data.area)
    
    y = t - x
    
    X = x.sum()
    Y = y.sum()
    T = t.sum()
    
    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()
    
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X/T) == False)[0][0]
    n2 = np.where(((np.cumsum(t[des_ind]) / T) < X/T) == False)[0][0]
    
    n  = data.shape[0]
    T1 = t[asc_ind][0:(n1+1)].sum()
    T2 = t[asc_ind][n2:n].sum()
    
    RCO = ((((x[asc_ind] * area[asc_ind] / X).sum()) / ((y[asc_ind] * area[asc_ind] / Y).sum())) - 1) / \
          ((((t[asc_ind] * area[asc_ind])[0:(n1+1)].sum() / T1) / ((t[asc_ind] * area[asc_ind])[n2:n].sum() / T2)) - 1)
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return RCO, core_data


class Relative_Concentration:
    """
    Calculation of Relative Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
       
    Examples
    --------
    In this example, we will calculate the relative concentration index (RCO) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_concentration_index = Relative_Concentration(gdf, 'nhblk10', 'pop10')
    >>> relative_concentration_index.statistic
    0.5364305924831142
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _relative_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_concentration
        
        
        
def _absolute_centralization(data, group_pop_var, total_pop_var):
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Absolute Centralization Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    area = np.array(data.area)
    
    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)
    
    center_lon = c_lons.mean()
    center_lat = c_lats.mean()
    
    X = x.sum()
    A = area.sum()

    center_dist = np.sqrt((c_lons - center_lon) ** 2 + (c_lats - center_lat) ** 2)
    
    asc_ind = center_dist.argsort() 
    
    Xi = np.cumsum(x[asc_ind]) / X
    Ai = np.cumsum(area[asc_ind]) / A
    
    ACE = np.nansum(shift(Xi, 1, cval=np.NaN) * Ai) - \
          np.nansum(Xi * shift(Ai, 1, cval=np.NaN))
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return ACE, core_data
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return ACE, core_data


class Absolute_Centralization:
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Absolute Centralization Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Examples
    --------
    In this example, we will calculate the absolute centralization index (ACE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_centralization_index = Absolute_Centralization(gdf, 'nhblk10', 'pop10')
    >>> absolute_centralization_index.statistic
    0.6416113799795511
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _absolute_centralization(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_centralization
        
        
        
def _relative_centralization(data, group_pop_var, total_pop_var):
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Centralization Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')

    y = t - x
    
    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)
    
    center_lon = c_lons.mean()
    center_lat = c_lats.mean()
    
    X = x.sum()
    Y = y.sum()

    center_dist = np.sqrt((c_lons - center_lon) ** 2 + (c_lats - center_lat) ** 2)
    
    asc_ind = center_dist.argsort() 
    
    Xi = np.cumsum(x[asc_ind]) / X
    Yi = np.cumsum(y[asc_ind]) / Y
    
    RCE = np.nansum(shift(Xi, 1, cval=np.NaN) * Yi) - \
          np.nansum(Xi * shift(Yi, 1, cval=np.NaN))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return RCE, core_data


class Relative_Centralization:
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Centralization Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the relative centralization index (RCE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_centralization_index = Relative_Centralization(gdf, 'nhblk10', 'pop10')
    >>> relative_centralization_index.statistic
    0.18550429720565376
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _relative_centralization(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_centralization
        
        
        
def _spatial_information_theory(data, group_pop_var, total_pop_var, w = None, unit_in_local_env = True, original_crs = {'init': 'epsg:4326'}):
    """
    Calculation of Spatial Information Theory index

    Parameters
    ----------

    data              : a geopandas DataFrame with a geometry column.
    
    group_pop_var     : string
                        The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var     : string
                        The name of variable in data that contains the total population of the unit
                    
    w                 : W
                        A PySAL weights object. If not provided, Queen contiguity matrix is used.
                        This is used to construct the local environment around each spatial unit.
    
    unit_in_local_env : boolean
                        A condition argument that states if the local environment around the unit comprises the unit itself. Default is True.
                        
    original_crs      : the original crs code given by a dict of data, but this is later be projected for the Mercator projection (EPSG = 3395).
                        This argument is also to avoid passing data without crs and, therefore, raising unusual results.
                        This index rely on the population density and we consider the area using squared kilometers. 

    Attributes
    ----------

    statistic : float
                Spatial Information Theory Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    This measure can be extended to a society with more than two groups.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if w is None:    
        w_object = Queen.from_dataframe(data)
    else:
        w_object = w
    
    if (not issubclass(type(w_object), pysal.lib.weights.W)):
        raise TypeError('w is not a PySAL weights object')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    data['compl_pop_var'] = data['total_pop_var'] - data['group_pop_var']
    
    
    # In this case, M = 2 according to Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    pi_1 = data['group_pop_var'].sum() / data['total_pop_var'].sum()
    pi_2 = data['compl_pop_var'].sum() / data['total_pop_var'].sum()
    E = -1 * (pi_1 * math.log(pi_1, 2) + pi_2 * math.log(pi_2, 2))
    T = data['total_pop_var'].sum()
    
    # Here you reproject the data using the Mercator projection
    data.crs = original_crs
    data = data.to_crs(crs = {'init': 'epsg:3395'})  # Mercator
    sqm_to_sqkm = 10 ** 6
    data['area_sq_km'] = data.area / sqm_to_sqkm
    tau_p = data['total_pop_var'] / data['area_sq_km']
    
    w_matrix = w_object.full()[0]

    if unit_in_local_env:
        np.fill_diagonal(w_matrix, 1)

    # The local context of each spatial unit is given by the aggregate context (this multiplication gives the local sum of each population)
    data['local_group_pop_var'] = np.matmul(data['group_pop_var'], w_matrix)
    data['local_compl_pop_var'] = np.matmul(data['compl_pop_var'], w_matrix)
    data['local_total_pop_var'] = np.matmul(data['total_pop_var'], w_matrix)
    
    pi_tilde_p_1 = np.array(data['local_group_pop_var'] / data['local_total_pop_var'])
    pi_tilde_p_2 = np.array(data['local_compl_pop_var'] / data['local_total_pop_var'])
    
    E_tilde_p = -1 * (pi_tilde_p_1 * np.log(pi_tilde_p_1) / np.log(2) + pi_tilde_p_2 * np.log(pi_tilde_p_2) / np.log(2))
    
    SIT = 1 - 1 / (T * E) * (tau_p * E_tilde_p).sum() # This is the H_Tilde according to Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SIT, core_data


class Spatial_Information_Theory:
    """
    Calculation of Spatial Information Theory index

    Parameters
    ----------

    data              : a geopandas DataFrame with a geometry column.
    
    group_pop_var     : string
                        The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var     : string
                        The name of variable in data that contains the total population of the unit
                    
    w                 : W
                        A PySAL weights object. If not provided, Queen contiguity matrix is used.
                        This is used to construct the local environment around each spatial unit.
    
    unit_in_local_env : boolean
                        A condition argument that states if the local environment around the unit comprises the unit itself. Default is True.

    original_crs      : the original crs code given by a dict of data, but this is later be projected for the Mercator projection (EPSG = 3395).
                        This argument is also to avoid passing data without crs and, therefore, raising unusual results.
                        This index rely on the population density and we consider the area using squared kilometers.

    Attributes
    ----------

    statistic : float
                Spatial Information Theory Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Examples
    --------
    In this example, we will calculate the degree of spatial information theory (SIT) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset. The neighborhood contiguity matrix is used.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10')
    >>> spatial_information_theory_index.statistic
    0.789200592777201
        
    To use different neighborhood matrices:
        
    >>> from pysal.lib.weights import Rook, KNN
    
    Assuming K-nearest neighbors with k = 4
    
    >>> knn = KNN.from_dataframe(gdf, k=4)
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10', w = knn)
    >>> spatial_information_theory_index.statistic
    0.7879736633559175
    
    Assuming Rook contiguity neighborhood
    
    >>> roo = Rook.from_dataframe(gdf)
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10', w = roo)
    >>> spatial_information_theory_index.statistic
    0.7891079229776317
            
    Notes
    -----
    Based on Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, w = None, unit_in_local_env = True):
        
        aux = _spatial_information_theory(data, group_pop_var, total_pop_var, w, unit_in_local_env)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_information_theory