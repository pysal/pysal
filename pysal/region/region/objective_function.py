from abc import ABC, abstractmethod

import itertools
import numpy as np

from region.util import get_metric_function


class ObjectiveFunction(ABC):
    def __init__(self, metric=None):
        """
        Parameters
        ----------
        metric : function or str or None, default: None
            Refer to the `metric` argument
            in :func:`region.util.get_metric_function`.
        """
        self.metric = get_metric_function(metric)

    @abstractmethod
    def __call__(self, labels, attr):
        """
        Calculate the objective value given the instance's current data.
        Parameters
        ----------
        labels : :class:`numpy.ndarray`
            The areas' region labels. Shape: number of areas.
        attr : :class:`numpy.ndarray`
            The areas' attributes. Shape: number of areas.

        Returns
        -------
        obj_val : float
            The objective value attained with the clustering defined by
            `labels`.
        """

    @abstractmethod
    def update(self, moving_area, recipient_region, labels, attr):
        """
        Calculate the difference in the objective value caused by moving
        `moving_area` to `recipient_region`.

        Parameters
        ----------
        moving_area : int
            The area to move.
        recipient_region : int
            The recipient region.
        labels : :class:`numpy.ndarray`
            The areas' region labels before the move. Shape: number of areas.
        attr : :class:`numpy.ndarray`
            The areas' attributes. Shape: number of areas.

        Returns
        -------
        diff : float
            The change in the objective function caused by moving `moving_area`
            to `recipient_region`."""


class ObjectiveFunctionPairwise(ObjectiveFunction):
    def __call__(self, labels, attr):
        """
        Examples
        --------
        >>> from sklearn.metrics.pairwise import distance_metrics
        >>> metric = distance_metrics()["manhattan"]
        >>> labels = np.array([0, 0, 0, 0, 1, 1])
        >>> attr = np.arange(len(labels)).reshape(-1, 1)
        >>> objective = ObjectiveFunctionPairwise(metric)
        >>> int(objective(labels, attr))
        11
        """
        regions_set = set(labels)
        obj_val = sum(self.metric(attr[i].reshape(1, -1),
                                  attr[j].reshape(1, -1))
                      for r in regions_set
                      for i, j in
                      itertools.combinations(np.where(labels == r)[0], 2))
        return obj_val

    def update(self, moving_area, recipient_region, labels, attr):
        print("update objective function pairwise")
        donor_region = labels[moving_area]

        attr_donor = attr[labels == donor_region]
        donor_diff = sum(self.metric(attr_donor,
                         attr[moving_area].reshape(1, -1)))

        attr_recipient = attr[labels == recipient_region]
        recipient_diff = sum(self.metric(attr_recipient,
                             attr[moving_area].reshape(1, -1)))
        print("obj. val. difference:", recipient_diff - donor_diff)
        return recipient_diff - donor_diff


class ObjectiveFunctionCenter(ObjectiveFunction):
    def __init__(self, metric=None, center=np.mean, reduction=np.sum):
        """
        Parameters
        ----------
        metric : function
            Refer to the corresponding argument in
            :meth:`ObjectiveFunction.__init__`.
        center : function, default: np.mean
            Function for calculating the center of the attributes of the areas
            belonging to the same region.
        reduction : function, default: np.sum
            Function used for

            * reducing the intraregional distances to a scalar and
            * reducing these scalars of the regions to one single scalar.
        """
        self.center = center
        self.reduction = reduction
        super().__init__(metric)

    def __call__(self, labels, attr):
        """
        Examples
        --------
        >>> from sklearn.metrics.pairwise import distance_metrics
        >>> metric = distance_metrics()["manhattan"]
        >>> labels = np.array([0, 1, 1])
        >>> attr = np.array([1, 2, 2]).reshape(-1, 1)
        >>> objective = ObjectiveFunctionCenter(metric)
        >>> int(objective(labels, attr))
        0
        >>> attr = np.array([1, 2, 3]).reshape(-1, 1)
        >>> objective = ObjectiveFunctionCenter(metric)
        >>> int(objective(labels, attr))
        1
        """
        regions = sorted(set(labels))
        objective_per_region = [self._intraregional_heterogeneity(labels, r,
                                                                  attr)
                                for r in regions]
        obj_val = self.reduction(objective_per_region)
        return obj_val

    def _intraregional_heterogeneity(self, labels, region, attr):
        return self.reduction(
                self.metric(
                        attr[labels == region],
                        self.center(attr[labels == region],
                                    axis=0).reshape(1, -1)),
                axis=0)

    def update(self, moving_area, recipient_region, labels, attr):
        donor_region = labels[moving_area]

        donor_before = self._intraregional_heterogeneity(
                labels, donor_region, attr)
        recipient_before = self._intraregional_heterogeneity(
                labels, recipient_region, attr)

        labels[moving_area] = recipient_region
        donor_after = self._intraregional_heterogeneity(
                labels, donor_region, attr)
        recipient_after = self._intraregional_heterogeneity(
                labels, recipient_region, attr)
        labels[moving_area] = donor_region

        overall_before = self.reduction((donor_before, recipient_before))
        overall_after = self.reduction((donor_after, recipient_after))
        diff = overall_after - overall_before

        print("diff", diff)
        print("labels", labels)
        return diff
