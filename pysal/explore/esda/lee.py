import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn import utils

class Spatial_Pearson(BaseEstimator):
    """Global Spatial Pearson Statistic"""

    def __init__(self, connectivity=None, permutations=999):
        """
        Initialize a spatial pearson estimator

        Arguments
        ---------
        connectivity:   scipy.sparse matrix object
                        the connectivity structure describing the relationships
                        between observed units. Will be row-standardized. 
        permutations:   int
                        the number of permutations to conduct for inference.
                        if < 1, no permutational inference will be conducted. 

        Attributes
        ----------
        association_: numpy.ndarray (2,2)
                      array containg the estimated Lee spatial pearson correlation
                      coefficients, where element [0,1] is the spatial correlation
                      coefficient, and elements [0,0] and [1,1] are the "spatial
                      smoothing factor"
        reference_distribution_: numpy.ndarray (n_permutations, 2,2)
                      distribution of correlation matrices for randomly-shuffled
                      maps. 
        significance_: numpy.ndarray (2,2)
                       permutation-based p-values for the fraction of times the
                       observed correlation was more extreme than the simulated 
                       correlations.
        """
        self.connectivity = connectivity
        self.permutations = permutations

    def fit(self, x, y):
        """
        bivariate spatial pearson's R based on Eq. 18 of Lee (2001).

        L = \dfrac{Z^T (V^TV) Z}{1^T (V^TV) 1}

        Arguments
        ---------
        x       :   numpy.ndarray
                    array containing continuous data
        y       :   numpy.ndarray
                    array containing continuous data

        Returns
        -------
        the fitted estimator.

        Notes
        -----
        Technical details and derivations can be found in :cite:`Lee2001`.

        """
        x = utils.check_array(x)
        y = utils.check_array(y)
        Z = numpy.column_stack((preprocessing.StandardScaler().fit_transform(x),
                                preprocessing.StandardScaler().fit_transform(y)))
        if self.connectivity is None:
            self.connectivity = sparse.eye(Z.shape[0])
        self.association_ = self._statistic(Z, self.connectivity) 
        
        standard_connectivity = sparse.csc_matrix(self.connectivity /
                                                  self.connectivity.sum(axis=1))

        if (self.permutations is None):
            return self
        elif self.permutations < 1:
            return self

        if self.permutations:
            simulations = [self._statistic(numpy.random.permutation(Z), self.connectivity)
                           for _ in range(self.permutations)]
            self.reference_distribution_ = simulations = numpy.array(simulations)
            above = simulations >= self.association_
            larger = above.sum(axis=0)
            extreme = numpy.minimum(self.permutations - larger, larger)
            self.significance_ = (extreme + 1.) / (self.permutations + 1.)
        return self

    @staticmethod
    def _statistic(Z,W):
        ctc = W.T @ W
        ones = numpy.ones(ctc.shape[0])
        return (Z.T @ ctc @ Z) / (ones.T @ ctc @ ones)


class Local_Spatial_Pearson(BaseEstimator):
    """Local Spatial Pearson Statistic"""

    def __init__(self, connectivity=None, permutations=999):
        """
        Initialize a spatial local pearson estimator

        Arguments
        ---------
        connectivity:   scipy.sparse matrix object
                        the connectivity structure describing the relationships
                        between observed units. Will be row-standardized. 
        permutations:   int
                        the number of permutations to conduct for inference.
                        if < 1, no permutational inference will be conducted. 
        significance_: numpy.ndarray (2,2)
                       permutation-based p-values for the fraction of times the
                       observed correlation was more extreme than the simulated 
                       correlations.
        Attributes
        ----------
        associations_: numpy.ndarray (n_samples,)
                      array containg the estimated Lee spatial pearson correlation
                      coefficients, where element [0,1] is the spatial correlation
                      coefficient, and elements [0,0] and [1,1] are the "spatial
                      smoothing factor"
        reference_distribution_: numpy.ndarray (n_permutations, n_samples)
                      distribution of correlation matrices for randomly-shuffled
                      maps. 
        significance_: numpy.ndarray (n_samples,)
                       permutation-based p-values for the fraction of times the
                       observed correlation was more extreme than the simulated 
                       correlations.


        Notes
        -----
        Technical details and derivations can be found in :cite:`Lee2001`.
        """
        self.connectivity = connectivity
        self.permutations = permutations

    def fit(self, x, y):
        """
        bivariate local pearson's R based on Eq. 22 in Lee (2001), using 
        site-wise conditional randomization from Moran_Local_BV.
        
        L_i = \dfrac{
                     n \cdot
                       \Big[\big(\sum_i w_{ij}(x_j - \bar{x})\big)
                            \big(\sum_i w_{ij}(y_j - \bar{y})\big) \Big]
                     } 
                    {
                     \sqrt{\sum_i (x_i - \bar{x})^2}
                     \sqrt{\sum_i (y_i - \bar{y})^2}}
            = \dfrac{
                     n \cdot
                       (\tilde{x}_j - \bar{x})
                       (\tilde{y}_j - \bar{y})
                     } 
                    {
                     \sqrt{\sum_i (x_i - \bar{x})^2}
                     \sqrt{\sum_i (y_i - \bar{y})^2}}

        Lee, Sang Il. (2001), "Developing a bivariate spatial 
        association measure: An integration of Pearson's r and 
        Moran's I." Journal of Geographical Systems, 3(4):369-385.

        Arguments
        ---------
        x       :   numpy.ndarray
                    array containing continuous data
        y       :   numpy.ndarray
                    array containing continuous data

        Returns
        -------
        the fitted estimator.
        """
        x = utils.check_array(x)
        x = preprocessing.StandardScaler().fit_transform(x)
        
        y = utils.check_array(y)
        y = preprocessing.StandardScaler().fit_transform(y)

        Z = numpy.column_stack((x, y))

        standard_connectivity = sparse.csc_matrix(self.connectivity /
                                                  self.connectivity.sum(axis=1))
        
        n, _ = x.shape

        self.associations_ = self._statistic(Z, standard_connectivity)

        if self.permutations:
            self.reference_distribution_ = numpy.empty((n, self.permutations))
            max_neighbors = (standard_connectivity != 0).sum(axis=1).max()
            random_ids = numpy.array([numpy.random.permutation(n - 1)[0:max_neighbors + 1]
                                      for i in range(self.permutations)])
            ids = numpy.arange(n)

            for i in range(n):
                row = standard_connectivity[i]
                weight = numpy.asarray(row[row.nonzero()]).reshape(-1,1)
                cardinality = row.nonzero()[0].shape[0]

                ids_not_i = ids[ids != i]
                numpy.random.shuffle(ids_not_i)
                randomizer = random_ids[:, 0:cardinality]
                random_neighbors = ids_not_i[randomizer]
                
                random_neighbor_x = x[random_neighbors]
                random_neighbor_y = y[random_neighbors]

                self.reference_distribution_[i] = (weight * random_neighbor_y - y.mean())\
                                                    .sum(axis=1).squeeze()
                self.reference_distribution_[i] *= (weight * random_neighbor_x - x.mean())\
                                                    .sum(axis=1).squeeze()
            above = self.reference_distribution_ >= self.associations_.reshape(-1,1)
            larger = above.sum(axis=1)
            extreme = numpy.minimum(larger, self.permutations - larger)
            self.significance_ = (extreme + 1.) / (self.permutations + 1.)
            self.reference_distribution_ = self.reference_distribution_.T
        else:
            self.reference_distribution_ = None
        return self

    @staticmethod
    def _statistic(Z,W):
        return (Z[:,1] @ W.T) * (W @ Z[:,0]) 

if __name__ == '__main__':
    import geopandas
    import pysal.lib
    df = geopandas.read_file(pysal.lib.examples.get_path('columbus.shp'))
    x = df[['HOVAL']].values
    y = df[['CRIME']].values
    zx = preprocessing.StandardScaler().fit_transform(x)
    zy = preprocessing.StandardScaler().fit_transform(y)
    w = pysal.lib.weights.Queen.from_dataframe(df)
    w.transform = 'r'
    numpy.random.seed(2478879)
    testglobal = Spatial_Pearson(connectivity=w.sparse).fit(x,y)
    numpy.random.seed(2478879)
    testlocal = Local_Spatial_Pearson(connectivity=w.sparse).fit(x,y)
