
import unittest
import pysal
import numpy as np
import random


class Test_Random_Regions(unittest.TestCase):
    def setUp(self):
        self.nregs = 13
        self.cards = range(2,14) + [10]
        self.w = pysal.lat2W(10,10,rook=False)
        self.ids = self.w.id_order
    
    def test_Random_Regions(self):
        random.seed(10)
        np.random.seed(10)
        t0 = pysal.region.Random_Regions(self.ids, permutations=2)
        result = [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])


        random.seed(60)
        np.random.seed(60)
        t0 = pysal.region.Random_Regions(self.ids, num_regions=self.nregs, cardinality=self.cards, contiguity=self.w, permutations=2)
        result = [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Regions(self.ids, num_regions=self.nregs, cardinality=self.cards, permutations=2)
        result = [37, 62]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Regions(self.ids, num_regions=self.nregs, contiguity=self.w, permutations=2)
        result = [71, 72, 70, 93, 51, 91, 85, 74, 63, 73, 61, 62, 82]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[1][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(60)
        np.random.seed(60)
        t0 = pysal.region.Random_Regions(self.ids, cardinality=self.cards, contiguity=self.w, permutations=2)
        result = [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Regions(self.ids, num_regions=self.nregs, permutations=2)
        result = [37, 62, 26, 41, 35, 25, 36]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Regions(self.ids, cardinality=self.cards, permutations=2)
        result = [37, 62]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[0][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Regions(self.ids, contiguity=self.w, permutations=2)
        result = [62, 52, 51, 50]
        for i in range(len(result)):
            self.assertEquals(t0.solutions[0].regions[1][i], result[i])
        for i in range(len(t0.solutions)):
            self.assertEquals(t0.solutions_feas[i], t0.solutions[i])
        
    def test_Random_Region(self):
        random.seed(10)
        np.random.seed(10)
        t0 = pysal.region.Random_Region(self.ids)
        t0.regions[0]
        result = [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(60)
        np.random.seed(60)
        t0 = pysal.region.Random_Region(self.ids, num_regions=self.nregs, cardinality=self.cards, contiguity=self.w)
        t0.regions[0]
        result = [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Region(self.ids, num_regions=self.nregs, cardinality=self.cards)
        t0.regions[0]
        result = [37, 62]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Region(self.ids, num_regions=self.nregs, contiguity=self.w)
        t0.regions[1]
        result = [71, 72, 70, 93, 51, 91, 85, 74, 63, 73, 61, 62, 82]
        for i in range(len(result)):
            self.assertEquals(t0.regions[1][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(60)
        np.random.seed(60)
        t0 = pysal.region.Random_Region(self.ids, cardinality=self.cards, contiguity=self.w)
        t0.regions[0]
        result = [88, 97, 98, 89, 99, 86, 78, 59, 49, 69, 68, 79, 77]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Region(self.ids, num_regions=self.nregs)
        t0.regions[0]
        result = [37, 62, 26, 41, 35, 25, 36]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Region(self.ids, cardinality=self.cards)
        t0.regions[0]
        result = [37, 62]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)
    
        random.seed(100)
        np.random.seed(100)
        t0 = pysal.region.Random_Region(self.ids, contiguity=self.w)
        t0.regions[0]
        result = [37, 27, 36, 17]
        for i in range(len(result)):
            self.assertEquals(t0.regions[0][i], result[i])
        self.assertEquals(t0.feasible, True)

        
suite = unittest.TestLoader().loadTestsFromTestCase(Test_Random_Regions)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

