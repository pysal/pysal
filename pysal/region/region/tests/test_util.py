from scipy.sparse import csr_matrix
import numpy as np
import pytest

from region.p_regions.azp import AZP
from region import util


def all_elements_equal(iterable):
    return all(iterable[0] == element for element in iterable)


def not_all_elements_equal(iterable):
    return not all_elements_equal(iterable)


def test_assert_feasible__pass_disconnected():
    adj = csr_matrix(np.array([[0, 0],
                               [0, 0]]))  # nodes 0 and 1 are not connected ...
    labels = np.array([0, 1])  # ... and assigned to different regions
    try:
        util.assert_feasible(labels, adj)
        util.assert_feasible(labels, adj, n_regions=2)
    except ValueError:
        pytest.fail()


def test_assert_feasible__pass_connected():
    adj = csr_matrix(np.array([[0, 1],
                               [1, 0]]))  # nodes 0 and 1 are connected ...
    labels = np.array([0, 0])  # ...and (case 1) assigned to the same region
    try:
        util.assert_feasible(labels, adj)
        util.assert_feasible(labels, adj, n_regions=1)
    except ValueError:
        pytest.fail()

    labels = np.array([0, 1])  # ...and (case 2) assigned to different regions
    try:
        util.assert_feasible(labels, adj)
        util.assert_feasible(labels, adj, n_regions=2)
    except ValueError:
        pytest.fail()


def test_assert_feasible__contiguity():
    with pytest.raises(ValueError) as exc_info:
        # nodes 0 and 1 are not connected ...
        adj = csr_matrix(np.array([[0, 0],
                                   [0, 0]]))
        # ... but assigned to the same region --> not feasible
        labels = np.array([0, 0])
        util.assert_feasible(labels, adj)
    assert "not spatially contiguous" in str(exc_info)


def test_assert_feasible__number_of_regions():
    with pytest.raises(ValueError) as exc_info:
        adj = csr_matrix(np.array([[0, 1],
                                   [1, 0]]))  # nodes 0 and 1 are connected ...
        labels = np.array([0, 0])  # ... and assigned to the same region
        # but this is not feasible under the condition n_regions = 2
        n_regions = 2
        util.assert_feasible(labels, adj, n_regions=n_regions)
    assert "The number of regions is" in str(exc_info)


def test_random_element_from():
    lst = list(range(1000))
    n_pops = 5
    popped = []
    for _ in range(n_pops):
        lst_copy = list(lst)
        popped.append(util.random_element_from(lst_copy))
        assert len(lst_copy) == len(lst)
    assert not_all_elements_equal(popped)


def test_pop_randomly_from():
    lst = list(range(1000))
    n_pops = 5
    popped = []
    for _ in range(n_pops):
        lst_copy = list(lst)
        popped.append(util.pop_randomly_from(lst_copy))
        assert len(lst_copy) == len(lst) - 1
    assert not_all_elements_equal(popped)


def test_AZP_azp_connected_component__one_area():
    adj = csr_matrix(np.array([[0]]))  # adjacency matrix for a single node
    azp = AZP()
    obtained = azp._azp_connected_component(adj,
                                            initial_clustering=np.array([0]),
                                            attr=np.array([123]))
    desired = np.array([0])
    assert obtained == desired
