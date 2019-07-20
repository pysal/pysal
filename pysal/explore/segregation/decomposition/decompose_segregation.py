"""
Decomposition Segregation based Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu>, and Sergio J. Rey <sergio.rey@ucr.edu>"


import warnings
from pysal.explore.segregation.util.util import _generate_counterfactual, _dep_message, DeprecationHelper

# Including old and new api in __all__ so users can use both

__all__ = ['Decompose_Segregation',
           'DecomposeSegregation']

# The Deprecation calls of the classes are located in the end of this script #

def _decompose_segregation(index1,
                           index2,
                           counterfactual_approach='composition'):
    """Decompose segregation differences into spatial and attribute components.

    Given two segregation indices of the same type, use Shapley decomposition
    to measure whether the differences between index measures arise from
    differences in spatial structure or population structure

    Parameters
    ----------
    index1 : segregation.SegIndex class
        First SegIndex class to compare.
    index2 : segregation.SegIndex class
        Second SegIndex class to compare.
    counterfactual_approach : str, one of
                              ["composition", "share", "dual_composition"]
        The technique used to generate the counterfactual population
        distributions.

    Returns
    -------
    tuple
        (shapley spatial component, 
         shapley attribute component, 
         core data of index1, 
         core data of index2, 
         data with counterfactual variables for index1, 
         data with counterfactual variables for index2)

    """
    df1 = index1.core_data.copy()
    df2 = index2.core_data.copy()

    assert index1._function == index2._function, "Segregation indices must be of the same type"

    counterfac_df1, counterfac_df2 = _generate_counterfactual(
        df1,
        df2,
        'group_pop_var',
        'total_pop_var',
        counterfactual_approach=counterfactual_approach)

    seg_func = index1._function

    # index for spatial 1, attribute 1
    G_S1_A1 = index1.statistic

    # index for spatial 2, attribute 2
    G_S2_A2 = index2.statistic

    # index for spatial 1 attribute 2 (counterfactual population for structure 1)
    G_S1_A2 = seg_func(counterfac_df1, 'counterfactual_group_pop',
                       'counterfactual_total_pop')[0]

    # index for spatial 2 attribute 1 (counterfactual population for structure 2)
    G_S2_A1 = seg_func(counterfac_df2, 'counterfactual_group_pop',
                       'counterfactual_total_pop')[0]

    # take the average difference in spatial structure, holding attributes constant
    C_S = 1 / 2 * (G_S1_A1 - G_S2_A1 + G_S1_A2 - G_S2_A2)

    # take the average difference in attributes, holding spatial structure constant
    C_A = 1 / 2 * (G_S1_A1 - G_S1_A2 + G_S2_A1 - G_S2_A2)

    return C_S, C_A, df1, df2, counterfac_df1, counterfac_df2, counterfactual_approach


class DecomposeSegregation:
    """Decompose segregation differences into spatial and attribute components.

    Given two segregation indices of the same type, use Shapley decomposition
    to measure whether the differences between index measures arise from
    differences in spatial structure or population structure

    Parameters
    ----------
    index1 : segregation.SegIndex class
        First SegIndex class to compare.
    index2 : segregation.SegIndex class
        Second SegIndex class to compare.
    counterfactual_approach : str, one of
                              ["composition", "share", "dual_composition"]
        The technique used to generate the counterfactual population
        distributions.

    Attributes
    ----------

    c_s : float
        Shapley's Spatial Component of the decomposition
                
    c_a : float
        Shapley's Attribute Component of the decomposition

    Methods
    ----------

    plot : Visualize features of the Decomposition performed
        plot_type : str, one of ['cdfs', 'maps']
        
        'cdfs' : visualize the cumulative distribution functions of the compositions/shares
        'maps' : visualize the spatial distributions for original data and counterfactuals generated and Shapley's components (only available for GeoDataFrames)

    Examples
    --------
    Several examples can be found at https://github.com/pysal/segregation/blob/master/notebooks/decomposition_wrapper_example.ipynb.
    
    """

    def __init__(self, index1, index2, counterfactual_approach='composition'):

        aux = _decompose_segregation(index1, index2, counterfactual_approach)

        self.c_s = aux[0]
        self.c_a = aux[1]
        self._df1 = aux[2]
        self._df2 = aux[3]
        self._counterfac_df1 = aux[4]
        self._counterfac_df2 = aux[5]
        self._counterfactual_approach = aux[6]

    def plot(self, plot_type='cdfs'):
        """
        Plot the Segregation Decomposition Profile
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib`')

        if (plot_type == 'cdfs'):
            if (self._counterfactual_approach == 'composition'):
                plt.suptitle(
                    'Spatial Component = {}, Attribute Component: {}'.format(
                        round(self.c_s, 3), round(self.c_a, 3)),
                    size=20)
                plt.step(
                    self._counterfac_df1['group_composition'].sort_values(),
                    self._counterfac_df1['group_composition'].rank(
                        pct=True).sort_values(),
                    label='First Context Group Composition')

                plt.step(
                    self._counterfac_df2['group_composition'].sort_values(),
                    self._counterfac_df2['group_composition'].rank(
                        pct=True).sort_values(),
                    label='Second Context Group Composition')
                plt.legend()

            if (self._counterfactual_approach == 'share'):
                plt.suptitle(
                    'Spatial Component = {}, Attribute Component: {}'.format(
                        round(self.c_s, 3), round(self.c_a, 3)),
                    size=20)
                plt.step((self._df1['group_pop_var'] /
                          self._df1['group_pop_var'].sum()).sort_values(),
                         (self._df1['group_pop_var'] /
                          self._df1['group_pop_var'].sum()).rank(
                              pct=True).sort_values(),
                         label='First Context Group Share')

                plt.step((self._df2['group_pop_var'] /
                          self._df2['group_pop_var'].sum()).sort_values(),
                         (self._df2['group_pop_var'] /
                          self._df2['group_pop_var'].sum()).rank(
                              pct=True).sort_values(),
                         label='Second Context Group Share')

                plt.step(
                    ((self._df1['total_pop_var'] - self._df1['group_pop_var'])
                     / (self._df1['total_pop_var'] -
                        self._df1['group_pop_var']).sum()).sort_values(),
                    ((self._df1['total_pop_var'] - self._df1['group_pop_var'])
                     / (self._df1['total_pop_var'] -
                        self._df1['group_pop_var']).sum()).rank(
                            pct=True).sort_values(),
                    label='First Context Complementary Group Share')

                plt.step(
                    ((self._df2['total_pop_var'] - self._df2['group_pop_var'])
                     / (self._df2['total_pop_var'] -
                        self._df2['group_pop_var']).sum()).sort_values(),
                    ((self._df2['total_pop_var'] - self._df2['group_pop_var'])
                     / (self._df2['total_pop_var'] -
                        self._df2['group_pop_var']).sum()).rank(
                            pct=True).sort_values(),
                    label='Second Context Complementary Group Share')
                plt.legend()

            if (self._counterfactual_approach == 'dual_composition'):
                plt.suptitle(
                    'Spatial Component = {}, Attribute Component: {}'.format(
                        round(self.c_s, 3), round(self.c_a, 3)),
                    size=20)
                plt.step(
                    self._counterfac_df1['group_composition'].sort_values(),
                    self._counterfac_df1['group_composition'].rank(
                        pct=True).sort_values(),
                    label='First Context Group Composition')

                plt.step(
                    self._counterfac_df2['group_composition'].sort_values(),
                    self._counterfac_df2['group_composition'].rank(
                        pct=True).sort_values(),
                    label='Second Context Group Composition')

                plt.step(
                    (1 -
                     self._counterfac_df1['group_composition']).sort_values(),
                    (1 - self._counterfac_df1['group_composition']).rank(
                        pct=True).sort_values(),
                    label='First Context Complementary Group Composition')

                plt.step(
                    (1 -
                     self._counterfac_df2['group_composition']).sort_values(),
                    (1 - self._counterfac_df2['group_composition']).rank(
                        pct=True).sort_values(),
                    label='Second Context Complementary Group Composition')

                plt.legend()

        if (plot_type == 'maps'):
            if (str(type(self._df1)) !=
                    '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
                raise TypeError(
                    'data is not a GeoDataFrame and, therefore, maps cannot be draw.'
                )

            # Subplots
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            fig.suptitle(
                'Spatial Component = {}, Attribute Component: {}'.format(
                    round(self.c_s, 3), round(self.c_a, 3)),
                size=20)
            fig.subplots_adjust(hspace=1.25, wspace=0.2,
                                top=0.95)  # hspace space between lines
            fig.tight_layout(rect=[
                0, 0, 1, 0.925
            ])  # rect is to position the suptitle above the subplots

            # Original First Context (Upper Left)
            self._counterfac_df1.plot(column='group_composition',
                                      cmap='OrRd',
                                      legend=True,
                                      ax=axs[0, 0])
            axs[0, 0].title.set_text('Original First Context Composition')
            axs[0, 0].axis('off')

            # Counterfactual First Context (Bottom Left)
            self._counterfac_df1.plot(column='counterfactual_composition',
                                      cmap='OrRd',
                                      legend=True,
                                      ax=axs[1, 0])
            axs[1, 0].title.set_text(
                'Counterfactual First Context Composition')
            axs[1, 0].axis('off')

            # Counterfactual Second Context (Upper Right)
            self._counterfac_df2.plot(column='counterfactual_composition',
                                      cmap='OrRd',
                                      legend=True,
                                      ax=axs[0, 1])
            axs[0, 1].title.set_text(
                'Counterfactual Second Context Composition')
            axs[0, 1].axis('off')

            # Original Second Context (Bottom Right)
            self._counterfac_df2.plot(column='group_composition',
                                      cmap='OrRd',
                                      legend=True,
                                      ax=axs[1, 1])
            axs[1, 1].title.set_text('Original Second Context Composition')
            axs[1, 1].axis('off')








# Deprecation Calls

msg = _dep_message("Decompose_Segregation", "DecomposeSegregation")
Decompose_Segregation = DeprecationHelper(DecomposeSegregation, message=msg)