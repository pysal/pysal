"""
Spatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import pysal.lib

from pysal.lib.weights import Queen, Kernel, lag_spatial
from pysal.lib.weights.util import fill_diagonal
from numpy import inf
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, haversine_distances
from scipy.ndimage.interpolation import shift

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix

from pysal.explore.segregation.aspatial.aspatial_indexes import _dissim, MinMax
from pysal.explore.segregation.aspatial.multigroup_aspatial_indexes import MultiInformationTheory, MultiDivergence
from pysal.explore.segregation.network import calc_access
from pysal.lib.weights.util import attach_islands

from pysal.explore.segregation.util.util import _dep_message, DeprecationHelper

# Including old and new api in __all__ so users can use both

__all__ = [

    'Spatial_Prox_Prof', 
    'SpatialProxProf',
    
    'Spatial_Dissim', 
    'SpatialDissim',
    
    'Boundary_Spatial_Dissim',
    'BoundarySpatialDissim',
    
    'Perimeter_Area_Ratio_Spatial_Dissim', 
    'PerimeterAreaRatioSpatialDissim',
    
    'SpatialMinMax',
    
    'Distance_Decay_Isolation',
    'DistanceDecayIsolation',
    
    'Distance_Decay_Exposure', 
    'DistanceDecayExposure', 
    
    'Spatial_Proximity', 
    'SpatialProximity',
    
    'Absolute_Clustering',
    'AbsoluteClustering',
    
    'Relative_Clustering', 
    'RelativeClustering', 
    
    'Delta', 
    
    'Absolute_Concentration',
    'AbsoluteConcentration',
    
    'Relative_Concentration', 
    'RelativeConcentration', 
    
    'Absolute_Centralization',
    'AbsoluteCentralization',
    
    'Relative_Centralization', 
    'RelativeCentralization', 
    
    'SpatialInformationTheory',
    'SpatialDivergence',

    'compute_segregation_profile'
]

# The Deprecation calls of the classes are located in the end of this script #

# suppress numpy divide by zero warnings because it occurs a lot during the
# calculation of many indices
np.seterr(divide='ignore', invalid='ignore')


def _build_local_environment(data, groups, w):
    """Convert observations into spatially-weighted sums.

    Parameters
    ----------
    data : DataFrame
        dataframe with local observations
    w : pysal.lib.weights object
        weights matrix defining the local environment

    Returns
    -------
    DataFrame
        Spatialized data

    """
    new_data = []
    w = fill_diagonal(w)
    for y in data[groups]:
        new_data.append(lag_spatial(w, data[y]))
    new_data = pd.DataFrame(dict(zip(groups, new_data)))
    return new_data


def _return_length_weighted_w(data):
    """
    Returns a PySAL weights object that the weights represent the length of the common boundary of two areal units that share border.
    Author: Levi Wolf <levi.john.wolf@gmail.com>. 
    Thank you, Levi!

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.

    Notes
    -----
    Currently it's not making any projection.

    """

    w = pysal.lib.weights.Rook.from_dataframe(
        data, ids=data.index.tolist(), geom_col=data._geometry_column_name)

    if (len(w.islands) == 0):
        w = w
    else:
        warnings('There are some islands in the GeoDataFrame.')
        w_aux = pysal.lib.weights.KNN.from_dataframe(
            data,
            ids=data.index.tolist(),
            geom_col=data._geometry_column_name,
            k=1)
        w = attach_islands(w, w_aux)

    adjlist = w.to_adjlist()
    islands = pd.DataFrame.from_records([{
        'focal': island,
        'neighbor': island,
        'weight': 0
    } for island in w.islands])
    merged = adjlist.merge(data.geometry.to_frame('geometry'), left_on='focal',
                           right_index=True, how='left')\
                    .merge(data.geometry.to_frame('geometry'), left_on='neighbor',
                           right_index=True, how='left', suffixes=("_focal", "_neighbor"))\

    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry='geometry_focal')
    merged['geometry_neighbor'] = gpd.GeoSeries(merged.geometry_neighbor)

    # Getting the shared boundaries
    merged['shared_boundary'] = merged.geometry_focal.intersection(
        merged.set_geometry('geometry_neighbor'))

    # Putting it back to a matrix
    merged['weight'] = merged.set_geometry('shared_boundary').length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = pysal.lib.weights.W.from_adjlist(
        merged_with_islands[['focal', 'neighbor', 'weight']])
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]

    length_weighted_w._reset()

    return length_weighted_w


def _spatial_prox_profile(data, group_pop_var, total_pop_var, m=1000):
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

    Returns
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.
    
    Reference: :cite:`hong2014measuring`.

    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(m) is not int):
        raise TypeError('m must be a string.')

    if (m < 2):
        raise ValueError('m must be greater than 1.')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    # Create the shortest distance path between two pair of units using Shimbel matrix. This step was well discussed in https://github.com/pysal/segregation/issues/5.
    w_libpysal = Queen.from_dataframe(data)
    graph = csr_matrix(w_libpysal.full()[0])
    delta = floyd_warshall(csgraph=graph, directed=False)

    def calculate_etat(t):
        g_t_i = np.where(data.group_pop_var / data.total_pop_var >= t, True,
                         False)
        k = g_t_i.sum()

        # i and j only varies in the units subset within the threshold in eta_t of Hong (2014).
        sub_delta_ij = delta[g_t_i, :][:, g_t_i]

        den = sub_delta_ij.sum()
        eta_t = (k**2 - k) / den
        return eta_t

    grid = np.linspace(0, 1, m)
    aux = np.array(list(map(calculate_etat, grid)))
    aux[aux == inf] = 0
    aux[aux == -inf] = 0
    curve = np.nan_to_num(aux, 0)

    threshold = data.group_pop_var.sum() / data.total_pop_var.sum()
    SPP = ((threshold - ((curve[grid < threshold]).sum() / m -
                         (curve[grid >= threshold]).sum() / m)) /
           (1 - threshold))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return SPP, grid, curve, core_data


class SpatialProxProf:
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
                Spatial Proximity Profile Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial proximity profile (SPP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import SpatialProxProf
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    >>> spat_prox_index = SpatialProxProf(gdf, 'nhblk10', 'pop10')
    >>> spat_prox_index.statistic
    0.11217269612149207
    
    You can plot the profile curve with the plot method.
    
    >>> spat_prox_index.plot()
        
    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.
    
    Reference: :cite:`hong2014measuring`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m=1000):

        aux = _spatial_prox_profile(data, group_pop_var, total_pop_var, m)

        self.statistic = aux[0]
        self.grid = aux[1]
        self.curve = aux[2]
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
        graph = plt.scatter(self.grid, self.curve, s=0.1)
        return graph


def _spatial_dissim(data,
                    group_pop_var,
                    total_pop_var,
                    w=None,
                    standardize=False):
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
        
    Returns
    ----------

    statistic : float
                Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.
    
    Reference: :cite:`morrill1991measure`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
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

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)

    if not standardize:
        cij = w_object.full()[0]
    else:
        cij = w_object.full()[0]
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

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


class SpatialDissim:
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import SpatialDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_dissim_index = SpatialDissim(gdf, 'nhblk10', 'pop10')
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
    
    Reference: :cite:`morrill1991measure`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 w=None,
                 standardize=False):

        aux = _spatial_dissim(data, group_pop_var, total_pop_var, w,
                              standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_dissim


def _boundary_spatial_dissim(data,
                             group_pop_var,
                             total_pop_var,
                             standardize=False):
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
        
    Returns
    ----------

    statistic : float
                Boundary Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.

    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        pi=np.where(data.total_pop_var == 0, 0, data.group_pop_var /
                    data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

    # manhattan_distances used to compute absolute distances
    num = np.multiply(manhattan_distances(data[['pi']]), cij).sum()
    den = cij.sum()
    BSD = D - num / den
    BSD

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return BSD, core_data


class BoundarySpatialDissim:
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import BoundarySpatialDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> boundary_spatial_dissim_index = BoundarySpatialDissim(gdf, 'nhblk10', 'pop10')
    >>> boundary_spatial_dissim_index.statistic
    0.28869903953453163
            
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize=False):

        aux = _boundary_spatial_dissim(data, group_pop_var, total_pop_var,
                                       standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _boundary_spatial_dissim


def _perimeter_area_ratio_spatial_dissim(data,
                                         group_pop_var,
                                         total_pop_var,
                                         standardize=True):
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

    Returns
    ----------

    statistic : float
                Perimeter/Area Ratio Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Originally based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    However, Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    points out that in Wong’s original there is an issue with the formula which is an extra division by 2 in the spatial interaction component.
    This function follows the formula present in the first Appendix of Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.

    References: :cite:`wong1993spatial` and :cite:`tivadar2019oasisr`.
        
    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        pi=np.where(data.total_pop_var == 0, 0, data.group_pop_var /
                    data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum()

    peri = data.length
    ai = data.area

    aux_sum = np.add(
        np.array(list((peri / ai))),
        np.array(list((peri / ai))).reshape((len(list((peri / ai))), 1)))

    max_pa = max(peri / ai)

    num = np.multiply(np.multiply(manhattan_distances(data[['pi']]), cij),
                      aux_sum).sum()
    den = 2 * max_pa

    PARD = D - (num / den)
    PARD

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return PARD, core_data


class PerimeterAreaRatioSpatialDissim:
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import PerimeterAreaRatioSpatialDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> perimeter_area_ratio_spatial_dissim_index = PerimeterAreaRatioSpatialDissim(gdf, 'nhblk10', 'pop10')
    >>> perimeter_area_ratio_spatial_dissim_index.statistic
    0.31260876347432687
            
    Notes
    -----
    Originally based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    However, Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    points out that in Wong’s original there is an issue with the formula which is an extra division by 2 in the spatial interaction component.
    This function follows the formula present in the first Appendix of Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    
    References: :cite:`wong1993spatial` and :cite:`tivadar2019oasisr`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize=True):

        aux = _perimeter_area_ratio_spatial_dissim(data, group_pop_var,
                                                   total_pop_var, standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _perimeter_area_ratio_spatial_dissim
        


class SpatialMinMax(MinMax):
    """Spatial MinMax Index.

    This class calculates the spatial version of the MinMax
    index. The data are "spatialized" by converting each observation
    to a "local environment" by creating a weighted sum of the focal unit with
    its neighboring observations, where the neighborhood is defined by a
    pysal.lib weights matrix or a pandana Network instance.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        geodataframe with
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    w   : pysal.lib.W
        distance-based PySAL spatial weights matrix instance
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_osm_network`
        or via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries.This is true by default, but if you plan to calculate several
        indices using the same network, then you can set this
        parameter to `False` to avoid precomputing repeatedly inside the
        function
        
    Attributes
    ----------

    statistic : float
                SpatialMinMax Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on O'Sullivan & Wong (2007). A Surface‐Based Approach to Measuring Spatial Segregation.
    Geographical Analysis 39 (2). https://doi.org/10.1111/j.1538-4632.2007.00699.x

    Reference: :cite:`osullivanwong2007surface`.
    
    We'd like to thank @AnttiHaerkoenen for this contribution!
    
    """

    def __init__(self, 
                 data, 
                 group_pop_var, 
                 total_pop_var,
                 network=None,
                 w=None,
                 decay='linear',
                 distance=2000,
                 precompute=True):
        
        data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                    total_pop_var: 'total_pop_var'})
    
        data['group_2_pop_var'] = data['total_pop_var'] - data['group_pop_var']
        
        groups = ['group_pop_var', 'group_2_pop_var']
        
        if w is None and network is None:
            points = [(p.x, p.y) for p in data.centroid]
            w = Kernel(points)

        if w and network:
            raise (
                "must pass either a pandana network or a pysal weights object\
                 but not both")
        elif network:
            df = calc_access(data,
                             variables=groups,
                             network=network,
                             distance=distance,
                             decay=decay,
                             precompute=precompute)
            groups = ["acc_" + group for group in groups]
        else:
            df = _build_local_environment(data, groups, w)
        
        df['resulting_total'] = df['group_pop_var'] + df['group_2_pop_var']
        
        super().__init__(df, 'group_pop_var', 'resulting_total')



def _distance_decay_isolation(data,
                              group_pop_var,
                              total_pop_var,
                              alpha=0.6,
                              beta=0.5,
                              metric='euclidean'):
    """
    Calculation of Distance Decay Isolation index

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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------

    statistic : float
                Distance Decay Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.

    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    X = x.sum()

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if (metric == 'euclidean'):
        dist = euclidean_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))

    if (metric == 'haversine'):
        dist = haversine_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))  # This needs to be latitude first!

    c = np.exp(-dist)
    
    if c.sum() < 10 ** (-15): 
        raise ValueError('It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to \'haversine\'')
    
    np.fill_diagonal(c, val = np.exp(-(alpha * data.area)**(beta)))
    
    Pij = np.multiply(c, t) / np.sum(np.multiply(c, t), axis=1)
        
    DDxPx = (np.array(x / X) *
             np.nansum(np.multiply(Pij, np.array(x / t)), axis=1)).sum()

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return DDxPx, core_data


class DistanceDecayIsolation:
    """
    Calculation of Distance Decay Isolation index

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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Attributes
    ----------

    statistic : float
                Distance Decay Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the distance decay isolation index (DDxPx) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import DistanceDecayIsolation
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_isolation_index = DistanceDecayIsolation(gdf, 'nhblk10', 'pop10')
    >>> spatial_isolation_index.statistic
    0.07214112078134231
            
    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 alpha=0.6,
                 beta=0.5,
                 metric='euclidean'):

        aux = _distance_decay_isolation(data, group_pop_var, total_pop_var,
                                        alpha, beta, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _distance_decay_isolation


def _distance_decay_exposure(data,
                             group_pop_var,
                             total_pop_var,
                             alpha=0.6,
                             beta=0.5,
                             metric='euclidean'):
    """
    Calculation of Distance Decay Exposure index

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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------

    statistic : float
                Distance Decay Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.

    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    y = t - x
    X = x.sum()

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if (metric == 'euclidean'):
        dist = euclidean_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))

    if (metric == 'haversine'):
        dist = haversine_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))  # This needs to be latitude first!

    c = np.exp(-dist)
    
    if c.sum() < 10 ** (-15): 
        raise ValueError('It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to \'haversine\'')
    
    np.fill_diagonal(c, val = np.exp(-(alpha * data.area)**(beta)))
    
    Pij = np.multiply(c, t) / np.sum(np.multiply(c, t), axis=1)
    
    DDxPy = (x / X * np.nansum(np.multiply(Pij, y / t), axis=1)).sum()

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return DDxPy, core_data


class DistanceDecayExposure:
    """
    Calculation of Distance Decay Exposure index

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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Attributes
    ----------

    statistic : float
                Distance Decay Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the distance decay exposure index (DDxPy) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import DistanceDecayExposure
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_exposure_index = DistanceDecayExposure(gdf, 'nhblk10', 'pop10')
    >>> spatial_exposure_index.statistic
    0.9605053172501217
            
    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 alpha=0.6,
                 beta=0.5,
                 metric='euclidean'):

        aux = _distance_decay_exposure(data, group_pop_var, total_pop_var,
                                       alpha, beta, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _distance_decay_exposure


def _spatial_proximity(data,
                       group_pop_var,
                       total_pop_var,
                       alpha=0.6,
                       beta=0.5,
                       metric='euclidean'):
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
    Returns
    ----------
    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    T = data.total_pop_var.sum()

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var,
                       ti=data.total_pop_var)

    X = data.xi.sum()
    Y = data.yi.sum()

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if (metric == 'euclidean'):
        dist = euclidean_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))

    if (metric == 'haversine'):
        dist = haversine_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))  # This needs to be latitude first!

    c = np.exp(-dist)
    
    if c.sum() < 10 ** (-15): 
        raise ValueError('It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to \'haversine\'')
    
    np.fill_diagonal(c, val = np.exp(-(alpha * data.area)**(beta)))
    
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    Ptt = ((np.array(data.ti) * c).T * np.array(data.ti)).sum() / T**2
    SP = (X * Pxx + Y * Pyy) / (T * Ptt)

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return SP, core_data


class SpatialProximity:
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import SpatialProximity
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_prox_index = SpatialProximity(gdf, 'nhblk10', 'pop10')
    >>> spatial_prox_index.statistic
    1.002191883006537
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 alpha=0.6,
                 beta=0.5,
                 metric='euclidean'):

        aux = _spatial_proximity(data, group_pop_var, total_pop_var, alpha,
                                 beta, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_proximity


def _absolute_clustering(data,
                         group_pop_var,
                         total_pop_var,
                         alpha=0.6,
                         beta=0.5,
                         metric='euclidean'):
    """
    Calculation of Absolute Clustering index
    
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
    Returns
    ----------
    statistic : float
                Absolute Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var)

    X = data.xi.sum()

    x = np.array(data.xi)
    t = np.array(data.total_pop_var)
    n = len(data)

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if (metric == 'euclidean'):
        dist = euclidean_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))

    if (metric == 'haversine'):
        dist = haversine_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))  # This needs to be latitude first!

    c = np.exp(-dist)
    
    if c.sum() < 10 ** (-15): 
        raise ValueError('It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to \'haversine\'')
    
    np.fill_diagonal(c, val = np.exp(-(alpha * data.area)**(beta)))
    
    ACL = ((((x/X) * (c * x).sum(axis = 1)).sum()) - ((X / n**2) * c.sum())) / \
          ((((x/X) * (c * t).sum(axis = 1)).sum()) - ((X / n**2) * c.sum()))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return ACL, core_data


class AbsoluteClustering:
    """
    Calculation of Absolute Clustering index
    
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
    Attributes
    ----------
    statistic : float
                Absolute Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the absolute clustering measure (ACL) for the Riverside County using the census tract data of 2010.
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
    
    >>> absolute_clust_index = Absolute_Clustering(gdf, 'nhblk10', 'pop10')
    >>> absolute_clust_index.statistic
    0.20979814508119624
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 alpha=0.6,
                 beta=0.5,
                 metric='euclidean'):

        aux = _absolute_clustering(data, group_pop_var, total_pop_var, alpha,
                                   beta, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_clustering


def _relative_clustering(data,
                         group_pop_var,
                         total_pop_var,
                         alpha=0.6,
                         beta=0.5,
                         metric='euclidean'):
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
    Returns
    ----------
    statistic : float
                Relative Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var)

    X = data.xi.sum()
    Y = data.yi.sum()

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if (metric == 'euclidean'):
        dist = euclidean_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))

    if (metric == 'haversine'):
        dist = haversine_distances(
            pd.DataFrame({
                'c_lats': c_lats,
                'c_lons': c_lons
            }))  # This needs to be latitude first!

    c = np.exp(-dist)
    
    if c.sum() < 10 ** (-15): 
        raise ValueError('It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to \'haversine\'')
    
    np.fill_diagonal(c, val = np.exp(-(alpha * data.area)**(beta)))
    
    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    RCL = Pxx / Pyy - 1
    
    if np.isnan(RCL):
        raise ValueError('It not possible to determine the distance between, at least, one pair of units. This is probably due to the magnitude of the number of the centroids. We recommend to reproject the geopandas DataFrame.')

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return RCL, core_data


class RelativeClustering:
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
                    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.
                    
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import RelativeClustering
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_clust_index = RelativeClustering(gdf, 'nhblk10', 'pop10')
    >>> relative_clust_index.statistic
    0.12418089857347714
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 alpha=0.6,
                 beta=0.5,
                 metric='euclidean'):

        aux = _relative_clustering(data, group_pop_var, total_pop_var, alpha,
                                   beta, metric)

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

    Returns
    ----------

    statistic : float
                Delta Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    X = x.sum()
    A = area.sum()

    DEL = 1 / 2 * abs(x / X - area / A).sum()

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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import Delta
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> delta_index = Delta(gdf, 'nhblk10', 'pop10')
    >>> delta_index.statistic
    0.8367330649317353
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.
    
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

    Returns
    ----------

    statistic : float
                Absolute Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    X = x.sum()
    T = t.sum()

    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()

    # A discussion about the extraction of n1 and n2 can be found in https://github.com/pysal/segregation/issues/43
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X / T) == False)[0][0] + 1
    n2_aux = np.where(((np.cumsum(t[des_ind]) / T) < X / T) == False)[0][0] + 1
    n2 = len(data) - n2_aux

    n = data.shape[0]
    T1 = t[asc_ind][0:n1].sum()
    T2 = t[asc_ind][n2:n].sum()

    ACO = 1- ((((x[asc_ind] * area[asc_ind] / X).sum()) - ((t[asc_ind] * area[asc_ind] / T1)[0:n1].sum())) / \
          (((t[asc_ind] * area[asc_ind] / T2)[n2:n].sum()) - ((t[asc_ind] * area[asc_ind]/T1)[0:n1].sum())))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return ACO, core_data


class AbsoluteConcentration:
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import AbsoluteConcentration
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_concentration_index = AbsoluteConcentration(gdf, 'nhblk10', 'pop10')
    >>> absolute_concentration_index.statistic
    0.9577607171503524
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

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

    Returns
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    y = t - x

    X = x.sum()
    Y = y.sum()
    T = t.sum()

    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()

    # A discussion about the extraction of n1 and n2 can be found in https://github.com/pysal/segregation/issues/43
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X / T) == False)[0][0] + 1
    n2_aux = np.where(((np.cumsum(t[des_ind]) / T) < X / T) == False)[0][0] + 1
    n2 = len(data) - n2_aux

    n = data.shape[0]
    T1 = t[asc_ind][0:n1].sum()
    T2 = t[asc_ind][n2:n].sum()

    RCO = ((((x[asc_ind] * area[asc_ind] / X).sum()) / ((y[asc_ind] * area[asc_ind] / Y).sum())) - 1) / \
          ((((t[asc_ind] * area[asc_ind])[0:n1].sum() / T1) / ((t[asc_ind] * area[asc_ind])[n2:n].sum() / T2)) - 1)

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return RCO, core_data


class RelativeConcentration:
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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import RelativeConcentration
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_concentration_index = RelativeConcentration(gdf, 'nhblk10', 'pop10')
    >>> relative_concentration_index.statistic
    0.5204046784837685
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _relative_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_concentration


def _absolute_centralization(data,
                             group_pop_var,
                             total_pop_var,
                             center="mean",
                             metric='euclidean'):
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------

    statistic     : float
                    Absolute Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
    
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if isinstance(center, str):
        if not center in [
                'mean', 'median', 'population_weighted_mean',
                'largest_population'
        ]:
            raise ValueError(
                'The center string must one of \'mean\', \'median\', \'population_weighted_mean\', \'largest_population\''
            )

        if (center == "mean"):
            center_lon = c_lons.mean()
            center_lat = c_lats.mean()

        if (center == "median"):
            center_lon = np.median(c_lons)
            center_lat = np.median(c_lats)

        if (center == "population_weighted_mean"):
            center_lon = np.average(c_lons, weights=t)
            center_lat = np.average(c_lats, weights=t)

        if (center == "largest_population"):
            center_lon = c_lons[np.where(t == t.max())].mean()
            center_lat = c_lats[np.where(t == t.max())].mean()

    if isinstance(center, tuple) or isinstance(center, list) or isinstance(
            center, np.ndarray):
        if np.array(center).shape != (2, ):
            raise ValueError('The center tuple/list/array must have length 2.')

        center_lon = center[0]
        center_lat = center[1]

    if isinstance(center, int):
        if (center > len(data) - 1) or (center < 0):
            raise ValueError('The center index must by in the range of data.')

        center_lon = data.iloc[[center]].centroid.x.values[0]
        center_lat = data.iloc[[center]].centroid.y.values[0]

    X = x.sum()
    A = area.sum()

    dlon = c_lons - center_lon
    dlat = c_lats - center_lat

    if (metric == 'euclidean'):
        center_dist = np.sqrt((dlon)**2 + (dlat)**2)

    if (metric == 'haversine'):
        center_dist = 2 * np.arcsin(np.sqrt(np.sin(dlat/2)**2 + np.cos(center_lat) * np.cos(c_lats) * np.sin(dlon/2)**2))
    
    if np.isnan(center_dist).sum() > 0:
        raise ValueError('It not possible to determine the center distance for, at least, one unit. This is probably due to the magnitude of the number of the centroids. We recommend to reproject the geopandas DataFrame.')
    
    asc_ind = center_dist.argsort()

    Xi = np.cumsum(x[asc_ind]) / X
    Ai = np.cumsum(area[asc_ind]) / A

    ACE = np.nansum(shift(Xi, 1, cval=np.NaN) * Ai) - \
          np.nansum(Xi * shift(Ai, 1, cval=np.NaN))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    center_values = [center_lon, center_lat]

    return ACE, core_data, center_values


class AbsoluteCentralization:
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    Attributes
    ----------

    statistic     : float
                    Absolute Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
                
    Examples
    --------
    In this example, we will calculate the absolute centralization index (ACE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import AbsoluteCentralization
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_centralization_index = AbsoluteCentralization(gdf, 'nhblk10', 'pop10')
    >>> absolute_centralization_index.statistic
    0.6416113799795511
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 center="mean",
                 metric='euclidean'):

        aux = _absolute_centralization(data, group_pop_var, total_pop_var,
                                       center, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self.center_values = aux[2]
        self._function = _absolute_centralization


def _relative_centralization(data,
                             group_pop_var,
                             total_pop_var,
                             center="mean",
                             metric='euclidean'):
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------

    statistic     : float
                    Relative Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.

    """
    
    if not metric in ['euclidean', 'haversine']:
        raise ValueError('metric must one of \'euclidean\', \'haversine\'')
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    y = t - x

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if isinstance(center, str):
        if not center in [
                'mean', 'median', 'population_weighted_mean',
                'largest_population'
        ]:
            raise ValueError(
                'The center string must one of \'mean\', \'median\', \'population_weighted_mean\', \'largest_population\''
            )

        if (center == "mean"):
            center_lon = c_lons.mean()
            center_lat = c_lats.mean()

        if (center == "median"):
            center_lon = np.median(c_lons)
            center_lat = np.median(c_lats)

        if (center == "population_weighted_mean"):
            center_lon = np.average(c_lons, weights=t)
            center_lat = np.average(c_lats, weights=t)

        if (center == "largest_population"):
            center_lon = c_lons[np.where(t == t.max())].mean()
            center_lat = c_lats[np.where(t == t.max())].mean()

    if isinstance(center, tuple) or isinstance(center, list) or isinstance(
            center, np.ndarray):
        if np.array(center).shape != (2, ):
            raise ValueError('The center tuple/list/array must have length 2.')

        center_lon = center[0]
        center_lat = center[1]

    if isinstance(center, int):
        if (center > len(data) - 1) or (center < 0):
            raise ValueError('The center index must by in the range of data.')

        center_lon = data.iloc[[center]].centroid.x.values[0]
        center_lat = data.iloc[[center]].centroid.y.values[0]

    X = x.sum()
    Y = y.sum()

    dlon = c_lons - center_lon
    dlat = c_lats - center_lat

    if (metric == 'euclidean'):
        center_dist = np.sqrt((dlon)**2 + (dlat)**2)

    if (metric == 'haversine'):
        center_dist = 2 * np.arcsin(
            np.sqrt(
                np.sin(dlat / 2)**2 +
                np.cos(center_lat) * np.cos(c_lats) * np.sin(dlon / 2)**2))

    if np.isnan(center_dist).sum() > 0:
        raise ValueError('It not possible to determine the center distance for, at least, one unit. This is probably due to the magnitude of the number of the centroids. We recommend to reproject the geopandas DataFrame.')

    asc_ind = center_dist.argsort()

    Xi = np.cumsum(x[asc_ind]) / X
    Yi = np.cumsum(y[asc_ind]) / Y

    RCE = np.nansum(shift(Xi, 1, cval=np.NaN) * Yi) - \
          np.nansum(Xi * shift(Yi, 1, cval=np.NaN))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    center_values = [center_lon, center_lat]

    return RCE, core_data, center_values


class RelativeCentralization:
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.
    
    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units. 
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Attributes
    ----------

    statistic     : float
                    Relative Centralization Index
            
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
        
    Examples
    --------
    In this example, we will calculate the relative centralization index (RCE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import pysal.explore.segregation
    >>> from pysal.explore.segregation.spatial import RelativeCentralization
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_centralization_index = RelativeCentralization(gdf, 'nhblk10', 'pop10')
    >>> relative_centralization_index.statistic
    0.18550429720565376
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 center="mean",
                 metric='euclidean'):

        aux = _relative_centralization(data, group_pop_var, total_pop_var,
                                       center, metric)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self.center_values = aux[2]
        self._function = _relative_centralization


class SpatialInformationTheory(MultiInformationTheory):
    """Spatial Multigroup Information Theory Index.

    This class calculates the spatial version of the multigroup information
    theory index. The data are "spatialized" by converting each observation
    to a "local environment" by creating a weighted sum of the focal unit with
    its neighboring observations, where the neighborhood is defined by a
    pysal.lib weights matrix or a pandana Network instance.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        geodataframe with
    groups : list
        list of columns on gdf representing population groups for which the SIT
        index should be calculated
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_osm_network`
        or via helper functions from OSMnet or UrbanAccess.
    w   : pysal.lib.W
        distance-based PySAL spatial weights matrix instance
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries.This is true by default, but if you plan to calculate several
        indices using the same network, then you can set this
        parameter to `False` to avoid precomputing repeatedly inside the
        function

    """

    def __init__(self,
                 data,
                 groups,
                 network=None,
                 w=None,
                 decay='linear',
                 distance=2000,
                 precompute=True):

        if w and network:
            raise (
                "must pass either a pandana network or a pysal weights object\
                 but not both")
        elif network:
            df = calc_access(data,
                             variables=groups,
                             network=network,
                             distance=distance,
                             decay=decay,
                             precompute=precompute)
            groups = ["acc_" + group for group in groups]
        else:
            df = _build_local_environment(data, groups, w)
        super().__init__(df, groups)


class SpatialDivergence(MultiDivergence):
    """Spatial Multigroup Divergence Index.

    This class calculates the spatial version of the multigroup divergence
    index. The data are "spatialized" by converting each observation
    to a "local environment" by creating a weighted sum of the focal unit with
    its neighboring observations, where the neighborhood is defined by a
    pysal.lib weights matrix or a pandana Network instance.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        geodataframe with
    groups : list
        list of columns on gdf representing population groups for which the
        divergence index should be calculated
    w   : pysal.lib.W
        distance-based PySAL spatial weights matrix instance
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_osm_network`
        or via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries.This is true by default, but if you plan to calculate several
        indices using the same network, then you can set this
        parameter to `False` to avoid precomputing repeatedly inside the
        function
    """

    def __init__(self,
                 data,
                 groups,
                 network=None,
                 w=None,
                 decay='linear',
                 distance=2000,
                 precompute=True):

        if w and network:
            raise (
                "must pass either a pandana network or a pysal weights object\
                 but not both")
        elif network:
            df = calc_access(data,
                             variables=groups,
                             network=network,
                             distance=distance,
                             decay=decay,
                             precompute=precompute)
            groups = ["acc_" + group for group in groups]
        else:
            df = _build_local_environment(data, groups, w)
        super().__init__(df, groups)


def compute_segregation_profile(gdf,
                                groups=None,
                                distances=None,
                                network=None,
                                decay='linear',
                                function='triangular',
                                precompute=True):
    """Compute multiscalar segregation profile.

    This function calculates several Spatial Information Theory indices with
    increasing distance parameters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe with rows as observations and columns as population
        variables. Note that if using a network distance, the coordinate
        system for this gdf should be 4326. If using euclidian distance,
        this must be projected into planar coordinates like state plane or UTM.
    groups : list
        list of variables .
    distances : list
        list of floats representing bandwidth distances that define a local
        environment.
    network : pandana.Network (optional)
        A pandana.Network likely created with
        `segregation.network.get_osm_network`.
    decay : str (optional)
        decay type to be used in pandana accessibility calculation (the
        default is 'linear').
    function: 'str' (optional)
        which weighting function should be passed to pysal.lib.weights.Kernel
        must be one of: 'triangular','uniform','quadratic','quartic','gaussian'
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries.This is true by default, but if you plan to calculate several
        segregation profiles using the same network, then you can set this
        parameter to `False` to avoid precomputing repeatedly inside the
        function

    Returns
    -------
    dict
        dictionary with distances as keys and SIT statistics as values

    Notes
    -----
    Based on Sean F. Reardon, Stephen A. Matthews, David O’Sullivan, Barrett A. Lee, Glenn Firebaugh, Chad R. Farrell, & Kendra Bischoff. (2008). The Geographic Scale of Metropolitan Racial Segregation. Demography, 45(3), 489–514. https://doi.org/10.1353/dem.0.0019.

    Reference: :cite:`Reardon2008`.

    """
    gdf = gdf.copy()
    gdf[groups] = gdf[groups].astype(float)
    indices = {}
    indices[0] = MultiInformationTheory(gdf, groups).statistic

    if network:
        if not gdf.crs['init'] == 'epsg:4326':
            gdf = gdf.to_crs(epsg=4326)
        groups2 = ['acc_' + group for group in groups]
        if precompute:
            maxdist = max(distances)
            network.precompute(maxdist)
        for distance in distances:
            distance = np.float(distance)
            access = calc_access(gdf,
                                 network,
                                 decay=decay,
                                 variables=groups,
                                 distance=distance,
                                 precompute=False)
            sit = MultiInformationTheory(access, groups2)
            indices[distance] = sit.statistic
    else:
        for distance in distances:
            w = Kernel.from_dataframe(gdf,
                                      bandwidth=distance,
                                      function=function)
            sit = SpatialInformationTheory(gdf, groups, w=w)
            indices[distance] = sit.statistic
    return indices


# Deprecation Calls

msg = _dep_message("Spatial_Prox_Prof", "SpatialProxProf")
Spatial_Prox_Prof = DeprecationHelper(SpatialProxProf, message=msg)

msg = _dep_message("Spatial_Dissim", "SpatialDissim")
Spatial_Dissim = DeprecationHelper(SpatialDissim, message=msg)

msg = _dep_message("Boundary_Spatial_Dissim", "BoundarySpatialDissim")
Boundary_Spatial_Dissim = DeprecationHelper(BoundarySpatialDissim, message=msg)

msg = _dep_message("Perimeter_Area_Ratio_Spatial_Dissim",
                   "PerimeterAreaRatioSpatialDissim")
Perimeter_Area_Ratio_Spatial_Dissim = DeprecationHelper(
    PerimeterAreaRatioSpatialDissim, message=msg)

msg = _dep_message("Distance_Decay_Isolation", "DistanceDecayIsolation")
Distance_Decay_Isolation = DeprecationHelper(DistanceDecayIsolation,
                                             message=msg)

msg = _dep_message("Distance_Decay_Exposure", "DistanceDecayExposure")
Distance_Decay_Exposure = DeprecationHelper(DistanceDecayExposure, message=msg)

msg = _dep_message("Spatial_Proximity", "SpatialProximity")
Spatial_Proximity = DeprecationHelper(SpatialProximity, message=msg)

msg = _dep_message("Absolute_Clustering", "AbsoluteClustering")
Absolute_Clustering = DeprecationHelper(AbsoluteClustering, message=msg)

msg = _dep_message("Relative_Clustering", "RelativeClustering")
Relative_Clustering = DeprecationHelper(RelativeClustering, message=msg)

msg = _dep_message("Absolute_Concentration", "AbsoluteConcentration")
Absolute_Concentration = DeprecationHelper(AbsoluteConcentration, message=msg)

msg = _dep_message("Relative_Concentration", "RelativeConcentration")
Relative_Concentration = DeprecationHelper(RelativeConcentration, message=msg)

msg = _dep_message("Absolute_Centralization", "AbsoluteCentralization")
Absolute_Centralization = DeprecationHelper(AbsoluteCentralization,
                                            message=msg)

msg = _dep_message("Relative_Centralization", "RelativeCentralization")
Relative_Centralization = DeprecationHelper(RelativeCentralization, message=msg)
