"""
Generate random regions

Randomly form regions given various types of constraints on cardinality and
composition.
"""

__author__ = "David Folch dfolch@fsu.edu, Serge Rey srey@asu.edu"

import numpy as np
from pysal.region.components import check_contiguity
from pysal.common import copy

__all__ = ["Random_Regions", "Random_Region"]


class Random_Regions:
    """Generate a list of Random_Region instances.

    Parameters
    ----------

    area_ids        : list
                      IDs indexing the areas to be grouped into regions (must
                      be in the same order as spatial weights matrix if this
                      is provided)

    num_regions     : integer
                      number of regions to generate (if None then this is
                      chosen randomly from 2 to n where n is the number of
                      areas)

    cardinality     : list
                      list containing the number of areas to assign to regions
                      (if num_regions is also provided then len(cardinality)
                      must equal num_regions; if cardinality=None then a list
                      of length num_regions will be generated randomly)

    contiguity      : W
                      spatial weights object (if None then contiguity will be
                      ignored)

    maxiter         : int
                      maximum number attempts (for each permutation) at finding
                      a feasible solution (only affects contiguity constrained
                      regions)

    compact         : boolean
                      attempt to build compact regions, note (only affects
                      contiguity constrained regions)

    max_swaps       : int
                      maximum number of swaps to find a feasible solution
                      (only affects contiguity constrained regions)

    permutations    : int
                      number of Random_Region instances to generate

    Attributes
    ----------

    solutions       : list
                      list of length permutations containing all Random_Region instances generated

    solutions_feas  : list
                      list of the Random_Region instances that resulted in feasible solutions

    Examples
    --------

    Setup the data

    >>> import random
    >>> import numpy as np
    >>> import pysal
    >>> nregs = 13
    >>> cards = range(2,14) + [10]
    >>> w = pysal.lat2W(10,10,rook=True)
    >>> ids = w.id_order

    Unconstrained

    >>> random.seed(10)
    >>> np.random.seed(10)
    >>> t0 = pysal.region.Random_Regions(ids, permutations=2)
    >>> t0.solutions[0].regions[0]
    [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]

    Cardinality and contiguity constrained (num_regions implied)

    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t1 = pysal.region.Random_Regions(ids, num_regions=nregs, cardinality=cards, contiguity=w, permutations=2)
    >>> t1.solutions[0].regions[0]
    [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]

    Cardinality constrained (num_regions implied)

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t2 = pysal.region.Random_Regions(ids, num_regions=nregs, cardinality=cards, permutations=2)
    >>> t2.solutions[0].regions[0]
    [37, 62]

    Number of regions and contiguity constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t3 = pysal.region.Random_Regions(ids, num_regions=nregs, contiguity=w, permutations=2)
    >>> t3.solutions[0].regions[1]
    [62, 52, 51, 63, 61, 73, 41, 53, 60, 83, 42, 31, 32]

    Cardinality and contiguity constrained

    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t4 = pysal.region.Random_Regions(ids, cardinality=cards, contiguity=w, permutations=2)
    >>> t4.solutions[0].regions[0]
    [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]

    Number of regions constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t5 = pysal.region.Random_Regions(ids, num_regions=nregs, permutations=2)
    >>> t5.solutions[0].regions[0]
    [37, 62, 26, 41, 35, 25, 36]

    Cardinality constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t6 = pysal.region.Random_Regions(ids, cardinality=cards, permutations=2)
    >>> t6.solutions[0].regions[0]
    [37, 62]

    Contiguity constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t7 = pysal.region.Random_Regions(ids, contiguity=w, permutations=2)
    >>> t7.solutions[0].regions[1]
    [62, 61, 71, 63]

    """
    def __init__(
        self, area_ids, num_regions=None, cardinality=None, contiguity=None,
        maxiter=100, compact=False, max_swaps=1000000, permutations=99):

        solutions = []
        for i in range(permutations):
            solutions.append(Random_Region(area_ids, num_regions, cardinality,
                                           contiguity, maxiter, compact, max_swaps))
        self.solutions = solutions
        self.solutions_feas = []
        for i in solutions:
            if i.feasible == True:
                self.solutions_feas.append(i)


class Random_Region:
    """Randomly combine a given set of areas into two or more regions based
    on various constraints.


    Parameters
    ----------

    area_ids        : list
                      IDs indexing the areas to be grouped into regions (must
                      be in the same order as spatial weights matrix if this
                      is provided)

    num_regions     : integer
                      number of regions to generate (if None then this is
                      chosen randomly from 2 to n where n is the number of
                      areas)

    cardinality     : list
                      list containing the number of areas to assign to regions
                      (if num_regions is also provided then len(cardinality)
                      must equal num_regions; if cardinality=None then a list
                      of length num_regions will be generated randomly)

    contiguity      : W
                      spatial weights object (if None then contiguity will be
                      ignored)

    maxiter         : int
                      maximum number attempts at finding a feasible solution
                      (only affects contiguity constrained regions)

    compact         : boolean
                      attempt to build compact regions (only affects
                      contiguity constrained regions)

    max_swaps       : int
                      maximum number of swaps to find a feasible solution
                      (only affects contiguity constrained regions)

    Attributes
    ----------

    feasible        : boolean
                      if True then solution was found

    regions         : list
                      list of lists of regions (each list has the ids of areas
                      in that region)

    Examples
    --------

    Setup the data

    >>> import random
    >>> import numpy as np
    >>> import pysal
    >>> nregs = 13
    >>> cards = range(2,14) + [10]
    >>> w = pysal.weights.lat2W(10,10,rook=True)
    >>> ids = w.id_order

    Unconstrained

    >>> random.seed(10)
    >>> np.random.seed(10)
    >>> t0 = pysal.region.Random_Region(ids)
    >>> t0.regions[0]
    [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]

    Cardinality and contiguity constrained (num_regions implied)

    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t1 = pysal.region.Random_Region(ids, num_regions=nregs, cardinality=cards, contiguity=w)
    >>> t1.regions[0]
    [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]

    Cardinality constrained (num_regions implied)

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t2 = pysal.region.Random_Region(ids, num_regions=nregs, cardinality=cards)
    >>> t2.regions[0]
    [37, 62]

    Number of regions and contiguity constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t3 = pysal.region.Random_Region(ids, num_regions=nregs, contiguity=w)
    >>> t3.regions[1]
    [62, 52, 51, 63, 61, 73, 41, 53, 60, 83, 42, 31, 32]

    Cardinality and contiguity constrained

    >>> random.seed(60)
    >>> np.random.seed(60)
    >>> t4 = pysal.region.Random_Region(ids, cardinality=cards, contiguity=w)
    >>> t4.regions[0]
    [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]

    Number of regions constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t5 = pysal.region.Random_Region(ids, num_regions=nregs)
    >>> t5.regions[0]
    [37, 62, 26, 41, 35, 25, 36]

    Cardinality constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t6 = pysal.region.Random_Region(ids, cardinality=cards)
    >>> t6.regions[0]
    [37, 62]

    Contiguity constrained

    >>> random.seed(100)
    >>> np.random.seed(100)
    >>> t7 = pysal.region.Random_Region(ids, contiguity=w)
    >>> t7.regions[0]
    [37, 36, 38, 39]

    """
    def __init__(
        self, area_ids, num_regions=None, cardinality=None, contiguity=None,
                    maxiter=1000, compact=False, max_swaps=1000000):

        self.n = len(area_ids)
        ids = copy.copy(area_ids)
        self.ids = list(np.random.permutation(ids))
        self.area_ids = area_ids
        self.regions = []
        self.feasible = True

        # tests for input argument consistency
        if cardinality:
            if self.n != sum(cardinality):
                self.feasible = False
                raise Exception('number of areas does not match cardinality')
        if contiguity:
            if area_ids != contiguity.id_order:
                self.feasible = False
                raise Exception('order of area_ids must match order in contiguity')
        if num_regions and cardinality:
            if num_regions != len(cardinality):
                self.feasible = False
                raise Exception('number of regions does not match cardinality')

        # dispatches the appropriate algorithm
        if num_regions and cardinality and contiguity:
            # conditioning on cardinality and contiguity (number of regions implied)
            self.build_contig_regions(num_regions, cardinality, contiguity,
                                      maxiter, compact, max_swaps)
        elif num_regions and cardinality:
            # conditioning on cardinality (number of regions implied)
            region_breaks = self.cards2breaks(cardinality)
            self.build_noncontig_regions(num_regions, region_breaks)
        elif num_regions and contiguity:
            # conditioning on number of regions and contiguity
            cards = self.get_cards(num_regions)
            self.build_contig_regions(num_regions, cards, contiguity,
                                      maxiter, compact, max_swaps)
        elif cardinality and contiguity:
            # conditioning on cardinality and contiguity
            num_regions = len(cardinality)
            self.build_contig_regions(num_regions, cardinality, contiguity,
                                      maxiter, compact, max_swaps)
        elif num_regions:
            # conditioning on number of regions only
            region_breaks = self.get_region_breaks(num_regions)
            self.build_noncontig_regions(num_regions, region_breaks)
        elif cardinality:
            # conditioning on number of cardinality only
            num_regions = len(cardinality)
            region_breaks = self.cards2breaks(cardinality)
            self.build_noncontig_regions(num_regions, region_breaks)
        elif contiguity:
            # conditioning on number of contiguity only
            num_regions = self.get_num_regions()
            cards = self.get_cards(num_regions)
            self.build_contig_regions(num_regions, cards, contiguity,
                                      maxiter, compact, max_swaps)
        else:
            # unconditioned
            num_regions = self.get_num_regions()
            region_breaks = self.get_region_breaks(num_regions)
            self.build_noncontig_regions(num_regions, region_breaks)

    def get_num_regions(self):
        return np.random.random_integers(2, self.n)

    def get_region_breaks(self, num_regions):
        region_breaks = set([])
        while len(region_breaks) < num_regions - 1:
            region_breaks.add(np.random.random_integers(1, self.n - 1))
        region_breaks = list(region_breaks)
        region_breaks.sort()
        return region_breaks

    def get_cards(self, num_regions):
        region_breaks = self.get_region_breaks(num_regions)
        cards = []
        start = 0
        for i in region_breaks:
            cards.append(i - start)
            start = i
        cards.append(self.n - start)
        return cards

    def cards2breaks(self, cards):
        region_breaks = []
        break_point = 0
        for i in cards:
            break_point += i
            region_breaks.append(break_point)
        region_breaks.pop()
        return region_breaks

    def build_noncontig_regions(self, num_regions, region_breaks):
        start = 0
        for i in region_breaks:
            self.regions.append(self.ids[start:i])
            start = i
        self.regions.append(self.ids[start:])

    def grow_compact(self, w, test_card, region, candidates, potential):
        # try to build a compact region by exhausting all existing
        # potential areas before adding new potential areas
        add_areas = []
        while potential and len(region) < test_card:
            pot_index = np.random.random_integers(0, len(potential) - 1)
            add_area = potential[pot_index]
            region.append(add_area)
            candidates.remove(add_area)
            potential.remove(add_area)
            add_areas.append(add_area)
        for i in add_areas:
            potential.extend([j for j in w.neighbors[i]
                                 if j not in region and
                                    j not in potential and
                                    j in candidates])
        return region, candidates, potential

    def grow_free(self, w, test_card, region, candidates, potential):
        # increment potential areas after each new area is
        # added to the region (faster than the grow_compact)
        pot_index = np.random.random_integers(0, len(potential) - 1)
        add_area = potential[pot_index]
        region.append(add_area)
        candidates.remove(add_area)
        potential.remove(add_area)
        potential.extend([i for i in w.neighbors[add_area]
                             if i not in region and
                                i not in potential and
                                i in candidates])
        return region, candidates, potential

    def build_contig_regions(self, num_regions, cardinality, w,
                                maxiter, compact, max_swaps):
        if compact:
            grow_region = self.grow_compact
        else:
            grow_region = self.grow_free
        iter = 0
        while iter < maxiter:

            # regionalization setup
            regions = []
            size_pre = 0
            counter = -1
            area2region = {}
            self.feasible = False
            swap_count = 0
            cards = copy.copy(cardinality)
            cards.sort()  # try to build largest regions first (pop from end of list)
            candidates = copy.copy(self.ids)  # these are already shuffled

            # begin building regions
            while candidates and swap_count < max_swaps:
                # setup test to determine if swapping is needed
                if size_pre == len(regions):
                    counter += 1
                else:
                    counter = 0
                    size_pre = len(regions)
                # test if swapping is needed
                if counter == len(candidates):

                    # start swapping
                    # swapping simply changes the candidate list
                    swap_in = None   # area to become new candidate
                    while swap_in is None:  # PEP8 E711
                        swap_count += 1
                        swap_out = candidates.pop(0)  # area to remove from candidates
                        swap_neighs = copy.copy(w.neighbors[swap_out])
                        swap_neighs = list(np.random.permutation(swap_neighs))
                        # select area to add to candidates (i.e. remove from an existing region)
                        for i in swap_neighs:
                            if i not in candidates:
                                join = i  # area linking swap_in to swap_out
                                swap_index = area2region[join]
                                swap_region = regions[swap_index]
                                swap_region = list(np.random.permutation(swap_region))
                                for j in swap_region:
                                    # test to ensure region connectivity after removing area
                                    swap_region_test = swap_region[:] + [swap_out]
                                    if check_contiguity(w, swap_region_test, j):
                                        swap_in = j
                                        break
                            if swap_in is not None:  # PEP8 E711
                                break
                        else:
                            candidates.append(swap_out)
                    # swapping cleanup
                    regions[swap_index].remove(swap_in)
                    regions[swap_index].append(swap_out)
                    area2region.pop(swap_in)
                    area2region[swap_out] = swap_index
                    candidates.append(swap_in)
                    counter = 0

                # setup to build a single region
                building = True
                seed = candidates.pop(0)
                region = [seed]
                potential = [i for i in w.neighbors[seed] if i in candidates]
                test_card = cards.pop()

                # begin building single region
                while building and len(region) < test_card:
                    if potential:
                        region, candidates, potential = grow_region(
                            w, test_card,
                                        region, candidates, potential)
                    else:
                        # not enough potential neighbors to reach test_card size
                        building = False
                        cards.append(test_card)
                        if len(region) in cards:
                            # constructed region matches another candidate region size
                            cards.remove(len(region))
                        else:
                            # constructed region doesn't match a candidate region size
                            candidates.extend(region)
                            region = []

                # cleanup when successful region built
                if region:
                    regions.append(region)
                    region_index = len(regions) - 1
                    for i in region:
                        area2region[i] = region_index   # area2region needed for swapping
            # handling of regionalization result
            if len(regions) < num_regions:
                # regionalization failed
                self.ids = list(np.random.permutation(self.ids))
                regions = []
                iter += 1
            else:
                # regionalization successful
                self.feasible = True
                iter = maxiter
        self.regions = regions

