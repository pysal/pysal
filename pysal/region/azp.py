"""
Automated Zoning Procedure

Assign n areas to k regions (k<<n) such that each region is composed of
spatially contiguous areas. 

Author(s):
    Serge Rey srey@asu.edu

References:
http://www.spatialanalysisonline.com/output/html/Districtingandre-districting.html



To Do:
    - add shape constraint
    - add ceiling constraint
    - parallelize
    
"""
import pysal
import numpy as num
import copy
from components import check_contiguity

LARGE=10**3 # number of initial solutions to try before failing
class Azp:
    """Autmated zoning procedure"""
    def __init__(self,w,z,k,floor=None,floor_variable=None,
                 ceiling=None,ceiling_variable=None,verbose=False):
        """
        Arguments:

            w: spatial weights object, simple contiguity

            z: n*m matrix of observations on m attributes across n areas. This
            is used to calculate intra-regional homogeneity

            k: number of regions to form

            floor: a minimum bound for a variable that has to be obtained
            in each region

            floor_variable: n*1 vector of observations on variable for the
            floor

            ceiling: an upper bound for a variable that has to be respected by
            each region

            ceiling_variable: n*1 vector of observations on variable for the
            ceiling

            max_it: the number of initial random solutions to try before
            concluding no initial solution can be found.

            verbose: if true debugging information is printed

            Note if floor is not set we take the minimum of the first variable
            in z as the floor. Similarly if ceiling is not set we
            take the maximum of the first variable  in z (times n) as the ceiling.
            Both of these will be non-binding.


        Attributes:

            area2region: mapping of areas to regions (dictionary with key being area
            id, value being region id.

            regions: sequence of region membership. List of lists of region
            definitions

        Methods:
            objective_function:
                calculates the value of the objective function given a current
                solution

            summary:
                prints summary information about the regions


        Notes:
            If no feasible solution is found a diagnostic message is printed.
            Exception is raised.

        Example
        =======

        >>> w=pysal.weights.weights.lat2gal()
        >>> z=range(w.n)
        >>> num.random.seed(10)
        >>> z=num.random.random_sample((w.n,2))
        >>> p=num.random.random(w.n)*100
        >>> k=4
        >>> floor=sum(p)/(k+1)
        >>> a=Azp(w,z,k,floor,floor_variable=p)
        >>> a.regions
        [[4, 9, 3, 8, 2, 14], [19, 13, 24, 18, 17], [20, 21, 16, 22, 11, 23], [7, 6, 1, 5, 0, 10, 15, 12]]
        """
        self.w=w
        self.z=z
        self.k=k
        self.verbose=verbose
        self.floor=floor
        self.floor_variable=floor_variable
        iters=0
        initial=True
        feasible=False
        attempts=0
        while initial:
            ids=range(w.n)
            ns=0
            num.random.shuffle(ids)
            neighbors=[]
            regions=[]
            area2region={}
            for r in range(k):
                region=ids[r]
                regions.append([region])
                area2region[region]=r
            self.regions=regions
            remaining=ids[k:]
            self.remaining=remaining
            self.area2region=area2region
            r=0
            for region in self.regions:
                r+=1
                building = True
                order=0
                while building:
                    # try to exhaust first order neighbors before rebuilding new
                    # neighbor list
                    neighbors=[]
                    added=False
                    for area in region:
                        candidates=w.neighbors[area]
                        candidates=[a for a in candidates if a in remaining]
                        neighbors.extend(candidates)
                    neighbors=set(neighbors)
                    for area in neighbors:
                        cv=sum(floor_variable[region])
                        if cv > self.floor:
                            building=False
                        else:
                            tmp=cv+floor_variable[area]
                            region.append(area)
                            self.area2region[area]=r
                            remaining.remove(area)
                            added=True
                    if not added:
                        building=False
                    order+=1
                        
            r=0
            for region in regions:
                fv=sum(floor_variable[region])
                if fv >= floor:
                    r+=1
            #raw_input('here')
            if r==k:
                initial=False
                feasible=True
                if self.verbose:
                    print 'Initial feasible solution found'
            if iters==LARGE:
                if self.verbose:
                    print 'no initial feasible solution found'
                raise NameError('Infeasible')
                initial=False
            if self.verbose:
                print 'initial attempt number: ',iters
            iters+=1
        if not feasible:
            pass
        else:
            self.regions=regions
            # assign any remaining areas
            # will have to check for any ceiling constraint
            while remaining:
                if self.verbose:
                    print 'len(remaining):',len(remaining)
                j=remaining.pop(0)
                candidate_regions=[]
                #check for contiguity constraint
                for region in self.regions:
                    for area in region:
                        if j in self.w.neighbors[area]:
                            candidate_regions.append(region)
                            break
                if not candidate_regions:
                    # no contiguities found yet so put back in que
                    if self.verbose:
                        print 'putting back on que: ',j
                    remaining.append(j)
                else:
                    num.random.shuffle(candidate_regions)
                    assign=candidate_regions[0]
                    id=self.regions.index(assign)
                    self.regions[id].append(j)
                
            # calculate internal homogeneity measure for all the clusters
            tss=self.objective_function()

            # rebuild area to region dictionary
            area2region={}
            for r,region in enumerate(regions):
                for area in region:
                    area2region[area]=r

            self.area2region=area2region
            # iteratively select a region and consider mergining areas on its
            # border 
            w=self.w
            regionIds=range(k)
            num.random.shuffle(regionIds)
            move_count=0
            while regionIds and move_count < LARGE:
                iternal_region=regionIds.pop()
                # find areas that are contiguous to iternal_region
                internal=self.regions[iternal_region]
                border=[]
                for i in internal:
                    for neighbor in w.neighbors[i]:
                        if neighbor not in internal and neighbor not in border:
                            # check if it would break contiguity
                            # or violates floor constraint
                            block=copy.copy(self.regions[area2region[neighbor]])

                            if check_contiguity(w,block,neighbor):
                                leave=copy.copy(self.regions[area2region[neighbor]])
                                #print block,leave
                                #ask=raw_input('here')
                                fv=sum(self.floor_variable[block])
                                if fv >= self.floor:
                                    border.append(neighbor)
                                """
                                tmp=block.remove(neighbor)
                                if sum(self.floor_variable[tmp]) >= self.floor:
                                    border.append(neighbor)
                                """
                internal_flag=True
                if not border:
                    internal_flag=False
                while internal_flag:
                    moves=[0]*len(border)
                    # bfs
                    for move,j in enumerate(border):
                        # calculate internal ss before swap
                        c_internal=copy.copy(internal)
                        c_border=copy.copy(self.regions[area2region[j]])
                        m_i=self.z[c_internal,:]
                        var=m_i.var(axis=0)
                        wss_i=sum(num.transpose(var)) * len(c_internal)
                        m_b=self.z[c_border,:]
                        var=m_b.var(axis=0)
                        wss_b=sum(num.transpose(var)) * len(c_border)
                        before=wss_i+wss_b
                        # temporarily swap j between border and internal
                        # region
                        c_internal.append(j)
                        c_border.remove(j)
                        m_i=self.z[c_internal,:]
                        var=m_i.var(axis=0)
                        wss_i=sum(num.transpose(var)) * len(c_internal)
                        m_b=self.z[c_border,:]
                        var=m_b.var(axis=0)
                        wss_b=sum(num.transpose(var)) * len(c_border)
                        after=wss_i+wss_b
                        moves[move]=after-before
                    try:
                        min_move=min(moves)
                    except:
                        min_move = 1.0
                    if min_move >= 0:
                        # no moves improve the solution
                        internal_flag=False
                    else:
                        # make the move and continue with this modified
                        # internal region
                        move_count+=1
                        j=moves.index(min(moves))
                        delta=moves[j]
                        j=border[j]
                        internal.append(j)
                        self.regions[area2region[j]].remove(j)
                        area2region[j]=iternal_region
                        self.regions[iternal_region]=internal
                        if self.verbose:
                            print '\nInternal region: ',iternal_region
                            print 'Swapping in area ',j
                            print 'Objective function: ',self.objective_function()
                            print 'Change in objective function: ',delta
                            print 'Number of moves: ',move_count
                        border=[]
                        for area in internal:
                            for neighbor in w.neighbors[area]:
                                if neighbor not in internal and neighbor not in border:
                                    block=copy.copy(self.regions[area2region[neighbor]])
                                    if check_contiguity(w,block,neighbor):
                                        fv=sum(self.floor_variable[block])
                                        if fv >= self.floor:
                                            border.append(neighbor)
            self.move_count=move_count


    def objective_function(self,solution=None):
        # solution is a list of lists of region ids [[1,7,2],[0,4,3],...] such
        # that the first region has areas 1,7,2 the second region 0,4,3 and so
        # on. solution does not have to be exhaustive
        if not solution:
            solution=self.regions
        wss=0
        for region in solution:
            m=self.z[region,:]
            var=m.var(axis=0)
            wss+=sum(num.transpose(var))*len(region)
        return wss

    def summary(self):
        print 'floor constraint: ',self.floor
        for region in self.regions:
            print sum(self.floor_variable[region])
        print 'ceiling constraint: ',self.ceiling
        for region in self.regions:
            print sum(self.ceiling_variable[region])

    def summary(self):
        for region in self.regions:
            print sum(self.floor_variable[region]),self.floor

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':

    _test()
