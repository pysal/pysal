"""
Max p regionalization

Heuristically form the maximum number (p) of regions given a set of n
areas and a floor constraint.
"""

__author__ = "Serge Rey <srey@asu.edu>, David Folch <dfolch@fsu.edu>"


import pysal
from components import check_contiguity
import copy
import numpy as np
from pysal.region import randomregion as RR

__all__ = ["Maxp", "Maxp_LISA"]

LARGE = 10 ** 6
MAX_ATTEMPTS = 100


class Maxp:
    """Try to find the maximum number of regions for a set of areas such that
    each region combines contiguous areas that satisfy a given threshold
    constraint.


    Parameters
    ----------

    w               : W
                      spatial weights object

    z               : array
                      n*m array of observations on m attributes across n
                      areas. This is used to calculate intra-regional
                      homogeneity
    floor           : int
                      a minimum bound for a variable that has to be
                      obtained in each region
    floor_variable  : array
                      n*1 vector of observations on variable for the floor
    initial         : int
                      number of initial solutions to generate
    verbose         : binary
                      if true debugging information is printed
    seeds           : list
                      ids of observations to form initial seeds. If
                      len(ids) is less than the number of observations, the
                      complementary ids are added to the end of seeds. Thus
                      the specified seeds get priority in the solution

    Attributes
    ----------

    area2region     : dict
                      mapping of areas to region. key is area id, value is
                      region id
    regions         : list
                      list of lists of regions (each list has the ids of areas
                      in that region)
    p               : int
                      number of regions
    swap_iterations : int
                      number of swap iterations
    total_moves     : int
                      number of moves into internal regions

    Examples
    --------

    Setup imports and set seeds for random number generators to insure the
    results are identical for each run.

    >>> import numpy as np
    >>> import pysal
    >>> np.random.seed(100)

    Setup a spatial weights matrix describing the connectivity of a square
    community with 100 areas.  Generate two random data attributes for each
    area in the community (a 100x2 array) called z. p is the data vector used to
    compute the floor for a region, and floor is the floor value; in this case
    p is simply a vector of ones and the floor is set to three. This means
    that each region will contain at least three areas.  In other cases the
    floor may be computed based on a minimum population count for example.

    >>> import numpy as np
    >>> import pysal
    >>> np.random.seed(100)
    >>> w = pysal.lat2W(10,10)
    >>> z = np.random.random_sample((w.n,2))
    >>> p = np.ones((w.n,1), float)
    >>> floor = 3
    >>> solution = pysal.region.Maxp(w, z, floor, floor_variable=p, initial=100)
    >>> solution.p
    29
    >>> min([len(region) for region in solution.regions])
    3
    >>> solution.regions[0]
    [76, 66, 56]
    >>>

    """
    def __init__(self, w, z, floor, floor_variable,
                 verbose=False, initial=100, seeds=[]):

        self.w = w
        self.z = z
        self.floor = floor
        self.floor_variable = floor_variable
        self.verbose = verbose
        self.seeds = seeds
        self.initial_solution()
        if not self.p:
            self.feasible = False
        else:
            self.feasible = True
            best_val = self.objective_function()
            self.current_regions = copy.copy(self.regions)
            self.current_area2region = copy.copy(self.area2region)
            self.initial_wss = []
            self.attempts = 0
            for i in range(initial):
                self.initial_solution()
                if self.p:
                    val = self.objective_function()
                    self.initial_wss.append(val)
                    if self.verbose:
                        print 'initial solution: ', i, val, best_val
                    if val < best_val:
                        self.current_regions = copy.copy(self.regions)
                        self.current_area2region = copy.copy(self.area2region)
                        best_val = val
                    self.attempts += 1
            self.regions = copy.copy(self.current_regions)
            self.p = len(self.regions)
            self.area2region = self.current_area2region
            if verbose:
                print "smallest region ifs: ", min([len(region) for region in self.regions])
                raw_input='wait'

            self.swap()

    def initial_solution(self):
        self.p = 0
        solving = True
        attempts = 0
        while solving and attempts <= MAX_ATTEMPTS:
            regions = []
            enclaves = []
            if not self.seeds:
                candidates = copy.copy(self.w.id_order)
                candidates = np.random.permutation(candidates)
                candidates = candidates.tolist()
            else:
                seeds = copy.copy(self.seeds)
                nonseeds = [i for i in self.w.id_order if i not in seeds]
                candidates = seeds
                candidates.extend(nonseeds)
            while candidates:
                seed = candidates.pop(0)
                # try to grow it till threshold constraint is satisfied
                region = [seed]
                building_region = True
                while building_region:
                    # check if floor is satisfied
                    if self.check_floor(region):
                        regions.append(region)
                        building_region = False
                    else:
                        potential = []
                        for area in region:
                            neighbors = self.w.neighbors[area]
                            neighbors = [neigh for neigh in neighbors if neigh in candidates]
                            neighbors = [neigh for neigh in neighbors if neigh not in region]
                            neighbors = [neigh for neigh in neighbors if neigh not in potential]
                            potential.extend(neighbors)
                        if potential:
                            # add a random neighbor
                            neigID = np.random.randint(0, len(potential))
                            neigAdd = potential.pop(neigID)
                            region.append(neigAdd)
                            # remove it from candidates
                            candidates.remove(neigAdd)
                        else:
                            #print 'enclave'
                            #print region
                            enclaves.extend(region)
                            building_region = False
            # check to see if any regions were made before going to enclave stage
            if regions:
                feasible = True
            else:
                attempts += 1
                break
            self.enclaves = enclaves[:]
            a2r = {}
            for r, region in enumerate(regions):
                for area in region:
                    a2r[area] = r
            encCount = len(enclaves)
            encAttempts = 0
            while enclaves and encAttempts != encCount:
                enclave = enclaves.pop(0)
                neighbors = self.w.neighbors[enclave]
                neighbors = [neighbor for neighbor in neighbors if neighbor not in enclaves]
                candidates = []
                for neighbor in neighbors:
                    region = a2r[neighbor]
                    if region not in candidates:
                        candidates.append(region)
                if candidates:
                    # add enclave to random region
                    regID = np.random.randint(0, len(candidates))
                    rid = candidates[regID]
                    regions[rid].append(enclave)
                    a2r[enclave] = rid
                    # structure to loop over enclaves until no more joining is possible
                    encCount = len(enclaves)
                    encAttempts = 0
                    feasible = True
                else:
                    # put back on que, no contiguous regions yet
                    enclaves.append(enclave)
                    encAttempts += 1
                    feasible = False
            if feasible:
                solving = False
                self.regions = regions
                self.area2region = a2r
                self.p = len(regions)
            else:
                if attempts == MAX_ATTEMPTS:
                    print 'No initial solution found'
                    self.p = 0
                attempts += 1

    def swap(self):
        swapping = True
        swap_iteration = 0
        if self.verbose:
            print 'Initial solution, objective function: ', self.objective_function()
        total_moves = 0
        self.k = len(self.regions)
        changed_regions = [1] * self.k
        nr = range(self.k)
        while swapping:
            moves_made = 0
            regionIds = [r for r in nr if changed_regions[r]]
            np.random.permutation(regionIds)
            changed_regions = [0] * self.k
            swap_iteration += 1
            for seed in regionIds:
                local_swapping = True
                local_attempts = 0
                while local_swapping:
                    local_moves = 0
                    # get neighbors
                    members = self.regions[seed]
                    neighbors = []
                    for member in members:
                        candidates = self.w.neighbors[member]
                        candidates = [candidate for candidate in candidates if candidate not in members]
                        candidates = [candidate for candidate in candidates if candidate not in neighbors]
                        neighbors.extend(candidates)
                    candidates = []
                    for neighbor in neighbors:
                        block = copy.copy(self.regions[self.area2region[
                            neighbor]])
                        if check_contiguity(self.w, block, neighbor):
                            block.remove(neighbor)
                            fv = self.check_floor(block)
                            if fv:
                                candidates.append(neighbor)
                    # find the best local move
                    if not candidates:
                        local_swapping = False
                    else:
                        nc = len(candidates)
                        moves = np.zeros([nc, 1], float)
                        best = None
                        cv = 0.0
                        for area in candidates:
                            current_internal = self.regions[seed]
                            current_outter = self.regions[self.area2region[
                                area]]
                            current = self.objective_function([current_internal, current_outter])
                            new_internal = copy.copy(current_internal)
                            new_outter = copy.copy(current_outter)
                            new_internal.append(area)
                            new_outter.remove(area)
                            new = self.objective_function([new_internal,
                                                           new_outter])
                            change = new - current
                            if change < cv:
                                best = area
                                cv = change
                        if best:
                            # make the move
                            area = best
                            old_region = self.area2region[area]
                            self.regions[old_region].remove(area)
                            self.area2region[area] = seed
                            self.regions[seed].append(area)
                            moves_made += 1
                            changed_regions[seed] = 1
                            changed_regions[old_region] = 1
                        else:
                            # no move improves the solution
                            local_swapping = False
                    local_attempts += 1
                    if self.verbose:
                        print 'swap_iteration: ', swap_iteration, 'moves_made: ', moves_made
                        print 'number of regions: ', len(self.regions)
                        print 'number of changed regions: ', sum(
                            changed_regions)
                        print 'internal region: ', seed, 'local_attempts: ', local_attempts
                        print 'objective function: ', self.objective_function()
                        print 'smallest region size: ',min([len(region) for region in self.regions])
            total_moves += moves_made
            if moves_made == 0:
                swapping = False
                self.swap_iterations = swap_iteration
                self.total_moves = total_moves
            if self.verbose:
                print 'moves_made: ', moves_made
                print 'objective function: ', self.objective_function()

    def check_floor(self, region):
        selectionIDs = [self.w.id_order.index(i) for i in region]
        cv = sum(self.floor_variable[selectionIDs])
        if cv >= self.floor:
            #print len(selectionIDs)
            return True
        else:
            return False

    def objective_function(self, solution=None):
        # solution is a list of lists of region ids [[1,7,2],[0,4,3],...] such
        # that the first region has areas 1,7,2 the second region 0,4,3 and so
        # on. solution does not have to be exhaustive
        if not solution:
            solution = self.regions
        wss = 0
        for region in solution:
            selectionIDs = [self.w.id_order.index(i) for i in region]
            m = self.z[selectionIDs, :]
            var = m.var(axis=0)
            wss += sum(np.transpose(var)) * len(region)
        return wss

    def inference(self, nperm=99):
        """Compare the within sum of squares for the solution against
        simulated solutions where areas are randomly assigned to regions that
        maintain the cardinality of the original solution.

        Parameters
        ----------

        nperm       : int
                      number of random permutations for calculation of
                      pseudo-p_values

        Attributes
        ----------

        pvalue      : float
                      pseudo p_value

        Examples
        --------

        Setup is the same as shown above except using a 5x5 community.

        >>> import numpy as np
        >>> import pysal
        >>> np.random.seed(100)
        >>> w=pysal.weights.lat2W(5,5)
        >>> z=np.random.random_sample((w.n,2))
        >>> p=np.ones((w.n,1),float)
        >>> floor=3
        >>> solution=pysal.region.Maxp(w,z,floor,floor_variable=p,initial=100)

        Set nperm to 9 meaning that 9 random regions are computed and used for
        the computation of a pseudo-p-value for the actual Max-p solution. In
        empirical work this would typically be set much higher, e.g. 999 or
        9999.

        >>> solution.inference(nperm=9)
        >>> solution.pvalue
        0.1

        """
        ids = self.w.id_order
        num_regions = len(self.regions)
        wsss = np.zeros(nperm + 1)
        self.wss = self.objective_function()
        cards = [len(i) for i in self.regions]
        sim_solutions = RR.Random_Regions(ids, num_regions,
                                          cardinality=cards, permutations=nperm)
        cv = 1
        c = 1
        for solution in sim_solutions.solutions_feas:
            wss = self.objective_function(solution.regions)
            wsss[c] = wss
            if wss <= self.wss:
                cv += 1
            c += 1
        self.pvalue = cv / (1. + len(sim_solutions.solutions_feas))
        self.wss_perm = wsss
        self.wss_perm[0] = self.wss

    def cinference(self, nperm=99, maxiter=1000):
        """Compare the within sum of squares for the solution against
        conditional simulated solutions where areas are randomly assigned to
        regions that maintain the cardinality of the original solution and
        respect contiguity relationships.

        Parameters
        ----------

        nperm       : int
                      number of random permutations for calculation of
                      pseudo-p_values

        maxiter     : int
                      maximum number of attempts to find each permutation

        Attributes
        ----------

        pvalue      : float
                      pseudo p_value

        feas_sols   : int
                      number of feasible solutions found

        Notes
        -----

        it is possible for the number of feasible solutions (feas_sols) to be
        less than the number of permutations requested (nperm); an exception
        is raised if this occurs.

        Examples
        --------

        Setup is the same as shown above except using a 5x5 community.

        >>> import numpy as np
        >>> import pysal
        >>> np.random.seed(100)
        >>> w=pysal.weights.lat2W(5,5)
        >>> z=np.random.random_sample((w.n,2))
        >>> p=np.ones((w.n,1),float)
        >>> floor=3
        >>> solution=pysal.region.Maxp(w,z,floor,floor_variable=p,initial=100)

        Set nperm to 9 meaning that 9 random regions are computed and used for
        the computation of a pseudo-p-value for the actual Max-p solution. In
        empirical work this would typically be set much higher, e.g. 999 or
        9999.

        >>> solution.cinference(nperm=9, maxiter=100)
        >>> solution.cpvalue
        0.1

        """
        ids = self.w.id_order
        num_regions = len(self.regions)
        wsss = np.zeros(nperm + 1)
        self.cwss = self.objective_function()
        cards = [len(i) for i in self.regions]
        sim_solutions = RR.Random_Regions(ids, num_regions,
                                          cardinality=cards, contiguity=self.w,
                                          maxiter=maxiter, permutations=nperm)
        self.cfeas_sols = len(sim_solutions.solutions_feas)
        if self.cfeas_sols < nperm:
            raise Exception('not enough feasible solutions found')
        cv = 1
        c = 1
        for solution in sim_solutions.solutions_feas:
            wss = self.objective_function(solution.regions)
            wsss[c] = wss
            if wss <= self.cwss:
                cv += 1
            c += 1
        self.cpvalue = cv / (1. + self.cfeas_sols)
        self.cwss_perm = wsss
        self.cwss_perm[0] = self.cwss


class Maxp_LISA(Maxp):
    """Max-p regionalization using LISA seeds

    Parameters
    ----------

    w              : W
                     spatial weights object
    z              : array
                     nxk array of n observations on k variables used to
                     measure similarity between areas within the regions.
    y              : array
                     nx1 array used to calculate the LISA statistics and
                     to set the intial seed order
    floor          : float
                     value that each region must obtain on floor_variable
    floor_variable : array
                     nx1 array of values for regional floor threshold
    initial        : int
                     number of initial feasible solutions to generate
                     prior to swapping

    Attributes
    ----------

    area2region     : dict
                      mapping of areas to region. key is area id, value is
                      region id
    regions         : list
                      list of lists of regions (each list has the ids of areas
                      in that region)
    swap_iterations : int
                      number of swap iterations
    total_moves     : int
                      number of moves into internal regions


    Notes
    -----

    We sort the observations based on the value of the LISAs. This
    ordering then gives the priority for seeds forming the p regions. The
    initial priority seeds are not guaranteed to be separated in the final
    solution.

    Examples
    --------

    Setup imports and set seeds for random number generators to insure the
    results are identical for each run.

    >>> import numpy as np
    >>> import pysal
    >>> np.random.seed(100)

    Setup a spatial weights matrix describing the connectivity of a square
    community with 100 areas.  Generate two random data attributes for each area
    in the community (a 100x2 array) called z. p is the data vector used to
    compute the floor for a region, and floor is the floor value; in this case
    p is simply a vector of ones and the floor is set to three. This means
    that each region will contain at least three areas.  In other cases the
    floor may be computed based on a minimum population count for example.

    >>> w=pysal.lat2W(10,10)
    >>> z=np.random.random_sample((w.n,2))
    >>> p=np.ones(w.n)
    >>> mpl=pysal.region.Maxp_LISA(w,z,p,floor=3,floor_variable=p)
    >>> mpl.p
    30
    >>> mpl.regions[0]
    [99, 98, 89]

    """
    def __init__(self, w, z, y, floor, floor_variable, initial=100):

        lis = pysal.Moran_Local(y, w)
        ids = np.argsort(lis.Is)
        ids = ids[range(w.n - 1, -1, -1)]
        ids = ids.tolist()
        mp = Maxp.__init__(
            self, w, z, floor=floor, floor_variable=floor_variable,
            initial=initial, seeds=ids)

