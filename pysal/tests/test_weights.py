import os
import unittest
import pysal

class _TestWeights(unittest.TestCase):
    def setUp(self):
        """ Setup the rtree and binning contiguity weights"""
        self.polyShp = '../examples/virginia.shp'
        self.pointShp = '../examples/juvenile.shp'
        self.w25 = pysal.lat2W(5,5)
    def test_iter(self):
        """ All methods names that begin with 'test' will be executed as a test case """
        self.assert_(os.path.exists(self.pointShp))
        w = pysal.rook_from_shapefile(self.polyShp)
        for i,j in zip(w,w):
            self.assertEquals(i,j)
    def test_rod(self):
        """ Make sure we can't write to the Read-Only Dictionary """
        self.assert_(os.path.exists(self.pointShp))
        w = pysal.rook_from_shapefile(self.polyShp)
        def set_rod():
            w.id2i[0] = 1
        self.assertRaises(TypeError,set_rod)
        
        
    def test_B(self):
        """ All methods names that begin with 'test' will be executed as a test case """
        self.assert_(os.path.exists(self.polyShp))

    def test_lat2W(self):
        self.assertEquals(self.w25.n,25)
        self.assertEquals(self.w25.s0,80.0)
        self.assertEquals(self.w25.s1,160.0)
        self.assertEquals(self.w25.s2array[23],36.)
        self.assertEquals(self.w25.trcW2,80.)
        self.assertEquals(self.w25.diagW2[23],3.)

    def test_w_f_shp(self):
        self.assert_(os.path.exists(self.pointShp))
        w = pysal.rook_from_shapefile(self.polyShp)
        self.assertEquals(w.n,136)
        self.assertEquals(w.s0,574.0)
        self.assertEquals(w.s1,1148.0)
        self.assertEquals(w.s2array[23],100.)
        self.assertEquals(w.trcW2,574.)
        self.assertEquals(w.diagW2[23],5.)
        self.assertEquals(w.diagWtW[23],5.)
        self.assertEquals(w.trcWtW,574.)
        self.assertEquals(w.diagWtW_WW[23],10.)
        self.assertEquals(w.trcWtW_WW,1148.)
        self.assertEquals(w.pct_nonzero,0.031033737024221453)
        self.assertEquals(w.cardinalities[2],4)
        self.assertEquals(w.max_neighbors,10)
        self.assertEquals(w.min_neighbors,1)
        self.assertEquals(w.nonzero,574)
        self.assertEquals(w.sd,2.2383396435442036)
        self.assertEquals(w.islands,[])
        self.assertEquals(w.asymmetries,[])
        self.assertEquals(w.transform,'O')
        self.assertEquals(w.weights[2],[1.0,1.0,1.0,1.0])
        w.transform='r'
        self.assertEquals(w.weights[2],[0.25,0.25,0.25,0.25])
        self.assertEquals(w.transform,'R')
        w.transform='v'
        self.assertEquals(w.weights[2],[0.25424173121040217, 0.25424173121040217, 0.25424173121040217, 0.25424173121040217])
        w.transform='b'
        self.assertEquals(w.weights[2],[1.0,1.0,1.0,1.0])
        self.assertFalse(w.asymmetry())
        w.transform='r'
        self.assertTrue(w.asymmetry())
        wf=w.full()[0]
        self.assertEquals(wf[2,0],0.25)
        self.assertEquals(wf[2,2],0)

    def test_util(self):
        w=self.w25
        shim = pysal.shimbel(w)
        self.assertEquals(shim[0][0:4],[-1,1,2,3])
        w8= pysal.higher_order(w, 8)
        self.assertEquals(w8.neighbors[0],[24])


suite = unittest.TestLoader().loadTestsFromTestCase(_TestWeights)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
