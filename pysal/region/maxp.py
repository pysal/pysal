"""
Max p regionalization

Heuristically form the maximum number (p) of regions given a set of n areas and a floor
constraint.

Author(s):
    Serge Rey srey@asu.edu

To Do:
    - add shape constraint
    - parallelize
    
"""
import pysal
from components import check_contiguity
from pysal.common import *

LARGE=10**6
MAX_ATTEMPTS=100

class Maxp:
    """Try to find the maximum number of regions for a set of areas such that
    each region combines continguous areas that satisfy a given threshold
    constraint."""
    def __init__(self,w,z,floor,floor_variable,verbose=False,initial=100):
        """
        Arguments:

            w: spatial weights object

            z: n*m matrix of observations on m attributes across n areas. This
            is used to calculate intra-regional homogeneity

            floor: a minimum bound for a variable that has to be obtained
            in each region

            floor_variable: n*1 vector of observations on variable for the
            floor

            initial: number of initial solutions to generate

            verbose: if true debugging information is printed

        Attributes:

            feasible: boolean identifying if a solution was found

            attempts: number of initial feasible solutions found

            area2region: mapping of areas to regions, dictionary with key being area
            id, value being region id, with region id index of region in
            regions.

            regions: sequence of region membership. List of lists of region
            definitions

            swap_iterations: number of swap iterations (passes through
            complete set of regions to do edge swapping)

            total_moves: number of moves into internal regions

        Methods:
            objective_function:
                calculates the value of the objective function given a current
                solution

        Example:
        >>> import random
        >>> import numpy as np
        >>> random.seed(100)
        >>> np.random.seed(100)
        >>> w=pysal.weights.weights.lat2gal(10,10)
        >>> z=np.random.random_sample((w.n,2))
        >>> p=np.random.random(w.n)*100
        >>> p=np.ones((w.n,1),float)
        >>> floor=3
        >>> solution=Maxp(w,z,floor,floor_variable=p,initial=100)
        >>> solution.p
        29
        >>> solution.regions[0]
        [99, 89, 79]
        >>> 
        """
    
        self.w=w
        self.z=z
        self.floor=floor
        self.floor_variable=floor_variable
        self.verbose=verbose
        self.initial_solution()
        if not self.p:
            self.feasible = False
        else:
            self.feasible = True 
            best_val=self.objective_function()
            self.current_regions=copy.copy(self.regions)
            self.current_area2region=copy.copy(self.area2region)
            self.initial_wss=[]
            self.attempts=0
            for i in range(initial):
                self.initial_solution()
                if self.p:
                    val=self.objective_function()
                    self.initial_wss.append(val)
                    if self.verbose:
                        print 'initial solution: ',i, val,best_val
                    if val < best_val:
                        self.current_regions=copy.copy(self.regions)
                        self.current_area2region=copy.copy(self.area2region)
                        best_val=val
                    self.attempts += 1
            self.regions=copy.copy(self.current_regions)
            self.area2region=self.current_area2region
            self.swap()

    def initial_solution(self):
        self.p=0
        solving=True
        attempts=0
        while solving and attempts<=MAX_ATTEMPTS:
            regions=[]
            enclaves=[]
            candidates=copy.copy(self.w.id_order)
            while candidates:
                id=random.randint(0,len(candidates)-1)
                seed=candidates.pop(id)
                # try to grow it till threshold constraint is satisfied
                region=[seed]
                building_region=True
                while building_region:
                    # check if floor is satisfied
                    if self.check_floor(region):
                        regions.append(region)
                        building_region=False
                    else:
                        potential=[] 
                        for area in region:
                            neighbors=self.w.neighbors[area]
                            neighbors=[neigh for neigh in neighbors if neigh in candidates]
                            neighbors=[neigh for neigh in neighbors if neigh not in region]
                            neighbors=[neigh for neigh in neighbors if neigh not in potential]
                            potential.extend(neighbors)
                        if potential:
                            # add a random neighbor
                            neigID=random.randint(0,len(potential)-1)
                            neigAdd=potential.pop(neigID)
                            region.append(neigAdd)
                            # remove it from candidates
                            candidates.remove(neigAdd)
                        else:
                            #print 'enclave'
                            #print region
                            enclaves.extend(region)
                            building_region=False
            # check to see if any regions were made before going to enclave stage
            if regions:
                feasible=True
            else:
                attempts+=1
                break
            self.enclaves=enclaves[:]
            a2r={}
            for r,region in enumerate(regions):
                for area in region:
                    a2r[area]=r
            encCount=len(enclaves)
            encAttempts=0
            while enclaves and encAttempts!=encCount:
                enclave=enclaves.pop(0)
                neighbors=self.w.neighbors[enclave]
                neighbors=[neighbor for neighbor in neighbors if neighbor not in enclaves]
                candidates=[]
                for neighbor in neighbors:
                    region=a2r[neighbor]
                    if region not in candidates:
                        candidates.append(region)
                if candidates:
                    # add enclave to random region
                    regID=random.randint(0,len(candidates)-1)
                    rid=candidates[regID]
                    regions[rid].append(enclave)
                    a2r[enclave]=rid
                    # structure to loop over enclaves until no more joining is possible
                    encCount=len(enclaves)
                    encAttempts=0
                    feasible=True
                else:
                    # put back on que, no contiguous regions yet
                    enclaves.append(enclave)
                    encAttempts+=1
                    feasible=False
            if feasible:
                solving=False
                self.regions=regions
                self.area2region=a2r
                self.p=len(regions)
            else:
                if attempts==MAX_ATTEMPTS:
                    print 'No initial solution found'
                    self.p=0
                attempts+=1

    def swap(self):
        swapping=True
        swap_iteration=0
        if self.verbose:
            print 'Initial solution, objective function: ',self.objective_function()
        total_moves=0
        self.k=len(self.regions)
        changed_regions=[1]*self.k
        nr=range(self.k)
        while swapping:
            moves_made=0
            regionIds=[r for r in nr if changed_regions[r]] 
            np.random.permutation(regionIds)
            changed_regions=[0]*self.k
            swap_iteration+=1
            for seed in regionIds:
                local_swapping=True
                local_attempts=0
                while local_swapping:
                    local_moves=0
                    # get neighbors
                    members=self.regions[seed]
                    neighbors=[]
                    for member in members:
                        candidates=self.w.neighbors[member]
                        candidates=[candidate for candidate in candidates if candidate not in members]
                        candidates=[candidate for candidate in candidates if candidate not in neighbors]
                        neighbors.extend(candidates)
                    candidates=[]
                    for neighbor in neighbors:
                        block=copy.copy(self.regions[self.area2region[neighbor]])
                        if check_contiguity(self.w,block,neighbor):
                            fv=self.check_floor(block)
                            if fv:
                                candidates.append(neighbor)
                    # find the best local move 
                    if not candidates:
                        local_swapping=False
                    else:
                        nc=len(candidates)
                        moves=np.zeros([nc,1],float)
                        best=None
                        cv=0.0
                        for area in candidates:
                            current_internal=self.regions[seed]
                            current_outter=self.regions[self.area2region[area]]
                            current=self.objective_function([current_internal,current_outter])
                            new_internal=copy.copy(current_internal)
                            new_outter=copy.copy(current_outter)
                            new_internal.append(area)
                            new_outter.remove(area)
                            new=self.objective_function([new_internal,new_outter])
                            change=new-current
                            if change < cv:
                                best=area
                                cv=change
                        if best:
                            # make the move
                            area=best
                            old_region=self.area2region[area]
                            self.regions[old_region].remove(area)
                            self.area2region[area]=seed
                            self.regions[seed].append(area)
                            moves_made+=1
                            changed_regions[seed]=1
                            changed_regions[old_region]=1
                        else:
                            # no move improves the solution
                            local_swapping=False
                    local_attempts+=1
                    if self.verbose:
                        print 'swap_iteration: ',swap_iteration,'moves_made: ',moves_made
                        print 'number of regions: ',len(self.regions)
                        print 'number of changed regions: ',sum(changed_regions)
                        print 'internal region: ',seed, 'local_attempts: ',local_attempts
                        print 'objective function: ',self.objective_function()
            total_moves+=moves_made
            if moves_made==0:
                swapping=False
                self.swap_iterations=swap_iteration
                self.total_moves=total_moves
            if self.verbose:
                print 'moves_made: ',moves_made
                print 'objective function: ',self.objective_function()

    def check_floor(self,region):                
        selectionIDs = [self.w.id_order.index(i) for i in region]
        cv=sum(self.floor_variable[selectionIDs])
        if cv >= self.floor:
            return True
        else:
            return False

    def objective_function(self,solution=None):
        # solution is a list of lists of region ids [[1,7,2],[0,4,3],...] such
        # that the first region has areas 1,7,2 the second region 0,4,3 and so
        # on. solution does not have to be exhaustive
        if not solution:
            solution=self.regions
        wss=0
        for region in solution:
            selectionIDs = [self.w.id_order.index(i) for i in region]
            m=self.z[selectionIDs,:]
            var=m.var(axis=0)
            wss+=sum(np.transpose(var))*len(region)
        return wss

    def inference(self,nperm=99):
        # compare the within sum of squares for the solution against nperm
        # solutions where areas are randomly assigned to regions
        ids=np.arange(self.w.n)
        regs=self.regions
        wsss=np.zeros(nperm+1)
        self.wss=self.objective_function()
        cv=1
        c=1
        for solution in range(nperm):
            ids=np.random.permutation(ids)
            r=[ids[reg] for reg in regs]
            wss=self.objective_function(r)
            wsss[c]=wss
            if wss<=self.wss:
                cv+=1
            c+=1
        self.pvalue=cv/(1.+nperm)
        self.wss_perm=wsss
        self.wss_perm[0]=self.wss

    def cinference(self,nperm=99,maxiter=100):
        # compare within sum of squares to conditional randomization that
        # respects contiguity cardinality distribution
        ns=0
        wss=np.zeros(nperm+1)
        regs=self.regions
        nregs=len(regs)
        w=copy.copy(self.w)
        n=w.n
        solutions=[]
        iter=0
        maxiter=nperm*maxiter
        while ns < nperm and iter<maxiter:
            solving=1
            nr=0
            candidates=range(self.w.n)
            seeds=np.random.permutation(n)[0:nregs]
            cards=[len(reg) for reg in self.regions]
            solution=[]
            regions=[]
            for seed in seeds:
                regions.append([seed])
                candidates.remove(seed)
            building=True
            flags=[1]*nregs
            while sum(flags):
                for r,region in enumerate(regions):
                    if flags[r]:
                        nr=len(region)
                        if nr in cards:
                            # done building this region
                            cards.remove(nr)
                            flags[r]=0
                        else:
                            # add a neighbor
                            neighbors=[]
                            for j in region:
                                neighbors.extend([ni for ni in w.neighbors[j] if ni in candidates])
                            if neighbors:
                                j=neighbors[0]
                                region.append(j)
                                candidates.remove(j)
                            else:
                                flags=[0]*nregs
                if not candidates:
                    t=[region.sort() for region in regions]
                    regions.sort()
                    if regions not in solutions:
                        solutions.append(regions)
                        ns+=1
            iter+=1
        self.csolutions=solutions
        self.citer=iter
        ids=np.arange(self.w.n)
        cwss=1
        self.wss=self.objective_function()
        for solution in solutions:
            wss=self.objective_function(solution)
            if wss<=self.wss:
                cwss+=1
        self.cpvalue=cwss*1. /( 1+len(solutions))


# tests

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
