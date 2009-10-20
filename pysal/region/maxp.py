
"""
Max p regionalization

Form the maximum number (p) of regions given a set of n areas and a floor
constraint.

Author(s):
    Serge Rey srey@asu.edu


Not to be used without permission of the author(s).


To Do:
    - add shape constraint
    - parallelize
    
"""
import random
import pysal
import numpy as num
from operator import gt, lt
import sys
import copy

LARGE=10**6
MAX_ATTEMPTS=100

class Maxp:
    """Maximum number of regions given a threshold constraint"""
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

            area2region: mapping of areas to regions, dictionary with key being area
            id, value being region id, with region id index of region in
            regions.

            attempts: number of initial attempts to find a feasible solution

            regions: sequence of region membership. List of lists of region
            definitions

            swap_iterations: number of swap iterations (passes through
            complete set of regions to do edge swapping)

            total_moves: number of moves into internal regions



        Methods:
            objective_function:
                calculates the value of the objective function given a current
                solution


        """
    
        self.w=w
        self.z=z
        self.floor=floor
        self.floor_variable=floor_variable
        self.verbose=verbose

        self.initial_solution()
        best_val=self.objective_function()
        self.current_regions=copy.copy(self.regions)
        self.current_area2region=copy.copy(self.area2region)
        for i in range(initial):
            self.initial_solution()
            val=self.objective_function()
            if self.verbose:
                print 'initial solution: ',i, val,best_val
            if val < best_val:
                self.current_regions=copy.copy(self.regions)
                self.current_area2region=copy.copy(self.area2region)
                best_val=val
        self.regions=copy.copy(self.current_regions)
        self.area2region=self.current_area2region

        self.swap()

    def initial_solution(self):
        solving=True
        attempts=0
        while solving and attempts<=MAX_ATTEMPTS:
            regions=[]
            enclaves=[]
            candidates=range(self.w.n)
            while candidates:
                id=random.randint(0,len(candidates)-1)
                seed=candidates.pop(id)
                # try to grow it till threshold constraint is satisfied
                region=[seed]
                building_region=True
                while building_region:
                    potential=[] 
                    for area in region:
                        neighbors=self.w.neighbors[area]
                        neighbors=[neigh for neigh in neighbors if neigh in candidates]
                        neighbors=[neigh for neigh in neighbors if neigh not in region]
                    if neighbors:
                        # add first neighbor
                        region.append(neighbors[0])
                        #  remove it from candidates
                        candidates.remove(neighbors[0])
                        # check if floor is satisfied
                        if self.check_floor(region):
                            regions.append(region)
                            building_region=False
                    else:
                        #print 'enclave'
                        #print region
                        enclaves.extend(region)
                        building_region=False
            self.enclaves=enclaves[:]
            a2r={}
            for r,region in enumerate(regions):
                for area in region:
                    a2r[area]=r
            repeat_enclave=[]
            feasible=True
            while enclaves:
                enclave=enclaves.pop(0)
                neighbors=self.w.neighbors[enclave]
                neighbors=[neighbor for neighbor in neighbors if neighbor not in enclaves]
                candidates=[]
                for neighbor in neighbors:
                    region=a2r[neighbor]
                    if region not in candidates:
                        candidates.append(region)
                if candidates:
                    # add first candidate to region
                    rid=candidates[0]
                    regions[rid].append(enclave)
                    a2r[enclave]=rid
                else:
                    # put back on que, no contiguous regions yet
                    if enclave in repeat_enclave:
                        enclaves=[]
                        feasible=False
                    else:
                        enclaves.append(enclave)
                        repeat_enclave.append(enclave)
            if feasible:
                solving=False
                self.regions=regions
                self.area2region=a2r
                solving=False
            else:
                if attempts==MAX_ATTEMPTS:
                    print 'No initial solution found'
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
            num.random.permutation(regionIds)
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
                        moves=num.zeros([nc,1],float)
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
        cv=sum(self.floor_variable[region])
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
            m=self.z[region,:]
            var=m.var(axis=0)
            wss+=sum(num.transpose(var))*len(region)
        return wss

        
def check_contiguity(w,neighbors,leaver):
    d={}
    g=Graph()
    for i in neighbors:
        d[i]=[j for j in w.neighbors[i] if (j in neighbors and j != leaver)]
    try:
        d.pop(leaver)
    except:
        pass
    for i in d:
        for j in d[i]:
            g.add_edge(i,j,1.0)
    cc=g.connected_components(op=gt)
    if len(cc)==1:
        neighbors.remove(leaver)
        if cc[0].nodes == set(neighbors):
            return True 
        else:
            return False
    else:
        return False

class Graph(object):
    def __init__(self):
        self.nodes=set()
        self.edges={}
        self.cluster_lookup={}
        self.no_link={}
    def add_edge(self,n1,n2,w):
        self.nodes.add(n1)
        self.nodes.add(n2)
        self.edges.setdefault(n1,{}).update({n2:w})
        self.edges.setdefault(n2,{}).update({n1:w})

    def connected_components(self,threshold=0.9, op=lt):
        nodes = set(self.nodes)
        components,visited =[], set()
        while len(nodes) > 0:
            connected, visited = self.dfs(nodes.pop(), visited, threshold, op)
            connected = set(connected)
            for node in connected:
                if node in nodes:
                    nodes.remove(node)

            subgraph=Graph()
            subgraph.nodes = connected
            subgraph.no_link = self.no_link
            for s in subgraph.nodes:
                for k,v in self.edges.get(s,{}).iteritems():
                    if k in subgraph.nodes:
                        subgraph.edges.setdefault(s,{}).update({k:v})
                if s in self.cluster_lookup:
                    subgraph.cluster_lookup[s] = self.cluster_lookup[s]
            components.append(subgraph)
        return components
    
    def dfs(self, v, visited, threshold, op=lt, first=None):
        aux=[v]
        visited.add(v)
        if first is None:
            first = v
        for i in (n for n, w in self.edges.get(v,{}).iteritems() \
                  if op(w, threshold) and n not in visited):
            x,y=self.dfs(i,visited,threshold,op,first)
            aux.extend(x)
            visited=visited.union(y)
        return aux, visited

         
if __name__ == '__main__':
    na=10
    w=pysal.weights.weights.lat2gal(na,na)
    z=range(w.n)
    random.seed(100)
    num.random.seed(100)
    z=num.random.random_sample((w.n,2))
    p=num.random.random(w.n)*100
    p=num.ones((w.n,1),float)
    floor=3
    solution=Maxp(w,z,floor,floor_variable=p,verbose=True,initial=10)
    solution2=Maxp(w,z,floor=50,floor_variable=p,verbose=True,initial=10)
    solution10=Maxp(w,z,floor=10,floor_variable=p,verbose=True,initial=10)




