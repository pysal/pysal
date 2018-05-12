import unittest as ut
import pysal as ps
import numpy as np
from sklearn.metrics import pairwise as skm
from .skater import Spanning_Forest
import types

import os
TESTDIR = os.path.dirname(os.path.abspath(__file__))


class Test_Skater(ut.TestCase):
    def setUp(self):
        self.df = ps.pdio.read_files(ps.examples.get_path('south.shp'))
        self.data = self.df[self.df.filter(like='90').columns.tolist() 
                  + self.df.filter(like='89').columns.tolist()].values
        self.data_c = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)
        self.W = ps.weights.Queen.from_dataframe(self.df)
        #self.south_5_q100 = np.load(os.path.join(TESTDIR, 'south_5_q500.ary'))
        self.south_inf_q20 = np.load(os.path.join(TESTDIR, 'south_inf_q20.ary'))
    
    def test_init(self):
        default = Spanning_Forest()
        self.assertEqual(default.metric, skm.manhattan_distances)
        self.assertEqual(default.center, np.mean)
        self.assertEqual(default.reduction, np.sum)
        change = Spanning_Forest(dissimilarity=skm.euclidean_distances,
                                 center=np.median, reduction=np.max)
        self.assertEqual(change.metric, skm.euclidean_distances)
        self.assertEqual(change.center, np.median)
        self.assertEqual(change.reduction, np.max)

        sym = Spanning_Forest(affinity=skm.cosine_similarity)
        self.assertIsInstance(sym.metric, types.LambdaType)
        test_distance = -np.log(skm.cosine_similarity(self.data[:2,]))
        comparator = sym.metric(self.data[:2,])
        np.testing.assert_allclose(test_distance, comparator)
    
    def test_run(self):
        result = Spanning_Forest().fit(5, self.W, self.data_c, quorum=50)
        
        #hmm... numbering is arbitrary... just check that they run for now
        #np.testing.assert_array_equal(self.south_5_q100, result.current_labels_)
        result2 = Spanning_Forest().fit(np.inf, self.W, self.data_c, quorum=20)
        
        #np.testing.assert_array_equal(self.south_inf_q20, result2.current_labels_)
