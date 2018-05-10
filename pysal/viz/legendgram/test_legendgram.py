import unittest as ut
import matplotlib.pyplot as plt
from pysal.contrib.viz.mapping import geoplot
import pysal as ps
from palettable import matplotlib as mplpal
from .util import inv_lut
from .legendgram import legendgram
import numpy as np

class Test_Legendgram(ut.TestCase):
    def setUp(self):
        self.data = ps.pdio.read_files(ps.examples.get_path('south.shp'))
        self.test_attribute = 'HR70'
        self.k = 10
        self.breaks = ps.Quantiles(self.data[self.test_attribute].values, k=self.k).bins
        self.pal = mplpal.Inferno_10

    def genframe(self):
        f,ax = plt.subplots()
        geoplot(self.data, self.test_attribute, classi='Quantiles',
                k=self.k, palette = mplpal.Inferno_10, ax=ax)
        return f,ax

    def test_call(self):
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values, 
                          self.breaks, mplpal.Inferno_10)
        plt.close(f)

    def test_positioning(self):
        """
        Check that changing the locstring changes the location of the plot. 
        Also, check that all strings & ints are able to be used. 
        """

        bboxes = []
        for i in range(1,11):
            f,ax = self.genframe()
            aout = legendgram(f,ax, self.data[self.test_attribute].values, 
                              self.breaks, mplpal.Inferno_10, loc=i)
            f2,ax2 = self.genframe()
            aout2 = legendgram(f2,ax2, self.data[self.test_attribute].values, 
                              self.breaks, mplpal.Inferno_10, loc=inv_lut[i])
            print(i,inv_lut[i])
            bbox = aout.get_position()
            bbox2 = aout2.get_position()
            print(bbox, bbox2)
            np.testing.assert_allclose(bbox.bounds, bbox2.bounds)
            bboxes.append(bbox)
            plt.close(f)
            plt.close(f2)
        for i in range(len(bboxes)-1):
            self.assertTrue(bboxes[i].bounds != bboxes[i+1].bounds)
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values, 
                              self.breaks, mplpal.Inferno_10, loc=0)
        bestbbox = aout.get_position()
        print(bestbbox.bounds, bboxes[2].bounds)
        np.testing.assert_allclose(bestbbox.bounds, bboxes[2].bounds) #best == bottom left

    def test_tickparams(self):
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values,
                          self.breaks, mplpal.Inferno_10, tick_params=dict(labelsize=20))
        ticks = aout.get_xticklabels()
        for tick in ticks:
            self.assertEqual(tick.get_fontsize(), 20)
        plt.close(f)

    def test_frameon(self):
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values,
                          self.breaks, mplpal.Inferno_10, frameon=True)
        self.assertTrue(aout.get_frame_on())
        plt.close(f)
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values,
                          self.breaks, mplpal.Inferno_10, frameon=False)
        self.assertTrue(not aout.get_frame_on())
        plt.close(f)

    @ut.skip('Not sure how to test this')
    def test_sizing(self):
        raise NotImplementedError('Not sure how to test this yet...')

    def test_clip(self):
        f,ax = self.genframe()
        aout = legendgram(f,ax, self.data[self.test_attribute].values,
                          self.breaks, mplpal.Inferno_10, clip=(10,20))
        self.assertEquals(aout.get_xlim(), (10,20))

    def test_palettebreak_mismatch(self):
        f,ax = self.genframe()
        with self.assertRaises(AssertionError):
            aout = legendgram(f,ax, self.data[self.test_attribute].values,
                              self.breaks, mplpal.Inferno_12)