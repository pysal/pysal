"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com>, Renan X. Cortes <renanc@ucr.edu>, and Eli Knaap <ek@knaaptime.com>"


import numpy as np
import math


def _generate_counterfactual(data1,
                             data2,
                             group_pop_var,
                             total_pop_var,
                             counterfactual_approach='composition'):
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
    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data1.columns)
            or (total_pop_var not in data1.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data1')

    if ((group_pop_var not in data2.columns)
            or (total_pop_var not in data2.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data2')

    if any(data1[total_pop_var] < data1[group_pop_var]):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units in data1.'
        )

    if any(data2[total_pop_var] < data2[group_pop_var]):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units in data2.'
        )

    df1 = data1.copy()
    df2 = data2.copy()

    if (counterfactual_approach == 'composition'):

        df1['group_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2[group_pop_var] / df2[total_pop_var])

        df1['counterfactual_group_pop'] = df1['group_composition'].rank(
            pct=True).apply(
                df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(
            pct=True).apply(
                df1['group_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_total_pop'] = df1[total_pop_var]
        df2['counterfactual_total_pop'] = df2[total_pop_var]

    if (counterfactual_approach == 'share'):

        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]

        df1['share'] = np.where(df1[total_pop_var] == 0, 0,
                                df1[group_pop_var] / df1[group_pop_var].sum())
        df2['share'] = np.where(df2[total_pop_var] == 0, 0,
                                df2[group_pop_var] / df2[group_pop_var].sum())

        df1['compl_share'] = np.where(
            df1['compl_pop_var'] == 0, 0,
            df1['compl_pop_var'] / df1['compl_pop_var'].sum())
        df2['compl_share'] = np.where(
            df2['compl_pop_var'] == 0, 0,
            df2['compl_pop_var'] / df2['compl_pop_var'].sum())

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_group = df1['share'].rank(pct=True).apply(
            df2['share'].quantile).sum()
        CT2_1_group = df2['share'].rank(pct=True).apply(
            df1['share'].quantile).sum()

        df1['counterfactual_group_pop'] = df1['share'].rank(pct=True).apply(
            df2['share'].quantile) / CT1_2_group * df1[group_pop_var].sum()
        df2['counterfactual_group_pop'] = df2['share'].rank(pct=True).apply(
            df1['share'].quantile) / CT2_1_group * df2[group_pop_var].sum()

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_compl = df1['compl_share'].rank(pct=True).apply(
            df2['compl_share'].quantile).sum()
        CT2_1_compl = df2['compl_share'].rank(pct=True).apply(
            df1['compl_share'].quantile).sum()

        df1['counterfactual_compl_pop'] = df1['compl_share'].rank(
            pct=True).apply(df2['compl_share'].quantile
                            ) / CT1_2_compl * df1['compl_pop_var'].sum()
        df2['counterfactual_compl_pop'] = df2['compl_share'].rank(
            pct=True).apply(df1['compl_share'].quantile
                            ) / CT2_1_compl * df2['compl_pop_var'].sum()

        df1['counterfactual_total_pop'] = df1[
            'counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2[
            'counterfactual_group_pop'] + df2['counterfactual_compl_pop']

    if (counterfactual_approach == 'dual_composition'):

        df1['group_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2[group_pop_var] / df2[total_pop_var])

        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]

        df1['compl_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1['compl_pop_var'] / df1[total_pop_var])
        df2['compl_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2['compl_pop_var'] / df2[total_pop_var])

        df1['counterfactual_group_pop'] = df1['group_composition'].rank(
            pct=True).apply(
                df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(
            pct=True).apply(
                df1['group_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_compl_pop'] = df1['compl_composition'].rank(
            pct=True).apply(
                df2['compl_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_compl_pop'] = df2['compl_composition'].rank(
            pct=True).apply(
                df1['compl_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_total_pop'] = df1[
            'counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2[
            'counterfactual_group_pop'] + df2['counterfactual_compl_pop']

    df1['group_composition'] = np.where(
        df1['total_pop_var'] == 0, 0,
        df1['group_pop_var'] / df1['total_pop_var'])
    df2['group_composition'] = np.where(
        df2['total_pop_var'] == 0, 0,
        df2['group_pop_var'] / df2['total_pop_var'])

    df1['counterfactual_composition'] = np.where(
        df1['counterfactual_total_pop'] == 0, 0,
        df1['counterfactual_group_pop'] / df1['counterfactual_total_pop'])
    df2['counterfactual_composition'] = np.where(
        df2['counterfactual_total_pop'] == 0, 0,
        df2['counterfactual_group_pop'] / df2['counterfactual_total_pop'])

    df1 = df1.drop(columns=['group_pop_var', 'total_pop_var'], axis=1)
    df2 = df2.drop(columns=['group_pop_var', 'total_pop_var'], axis=1)

    return df1, df2


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """Reproject gdf into the appropriate UTM zone.

    Project a GeoDataFrame to the UTM zone appropriate for its geometries'
    centroid.
    The simple calculation in this function works well for most latitudes, but
    won't work for some far northern locations like Svalbard and parts of far
    northern Norway.

    This function is lovingly modified from osmnx:
    https://github.com/gboeing/osmnx/

    Parameters
    ----------
    gdf : GeoDataFrame
        the gdf to be projected
    to_crs : dict
        if not None, just project to this CRS instead of to UTM
    to_latlong : bool
        if True, projects to latlong instead of to UTM

    Returns
    -------
    GeoDataFrame

    """
    assert len(gdf) > 0, 'You cannot project an empty GeoDataFrame.'

    # else, project the gdf to UTM
    # if GeoDataFrame is already in UTM, just return it
    if (gdf.crs is not None) and ('+proj=utm ' in gdf.crs):
        return gdf

    # calculate the centroid of the union of all the geometries in the
    # GeoDataFrame
    avg_longitude = gdf['geometry'].unary_union.centroid.x

    # calculate the UTM zone from this avg longitude and define the UTM
    # CRS to project
    utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1)
    utm_crs = '+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(
        utm_zone)

    # project the GeoDataFrame to the UTM CRS
    projected_gdf = gdf.to_crs(utm_crs)

    return projected_gdf


        
def _dep_message(original, replacement, when="2020-01-31", version="2.1.0"):
    msg = "Deprecated (%s): %s" % (version, original)
    msg += " is being renamed to %s." % replacement
    msg += " %s will be removed on %s." % (original, when)
    return msg

class DeprecationHelper(object):
    def __init__(self, new_target, message="Deprecated"):
        self.new_target = new_target
        self.message = message

    def _warn(self):
        from warnings import warn

        warn(self.message)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)