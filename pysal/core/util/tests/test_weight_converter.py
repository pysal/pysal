import unittest
import pysal
from pysal.core.util.weight_converter import WeightConverter
import tempfile
import os
import warnings

class test_WeightConverter(unittest.TestCase):
    def setUp(self):
        self.base_dir = '../../../examples/'
        self.test_files = ['arcgis_ohio.dbf', 'arcgis_txt.txt', 'ohio.swm',
                           'wmat.dat','wmat.mtx','sids2.gal','juvenile.gwt',
                           'geobugs_scot','stata_full.txt','stata_sparse.txt',
                           'spat-sym-us.mat','spat-sym-us.wk1']
        dataformats = ['arcgis_dbf','arcgis_text',None,None,None,None,None,
                       'geobugs_text','stata_text','stata_text',None,None]
        ns = [88, 3, 88, 49, 49, 100, 168, 56, 56, 56, 46, 46]
        self.dataformats = dict(zip(self.test_files,dataformats))
        self.ns = dict(zip(self.test_files,ns))
        self.fileformats = [('dbf','arcgis_dbf'),('txt','arcgis_text'),('swm',None),
                            ('dat',None),('mtx',None),('gal',None),('','geobugs_text'),
                            ('txt','stata_text'),('mat',None),('wk1',None)]

    def test__setW(self):
        for f in self.test_files:
            with warnings.catch_warnings(record=True) as warn:
                # note: we are just suppressing the warnings here; individual warnings 
                #       are tested in their specific readers
                warnings.simplefilter("always")
                wc = WeightConverter(self.base_dir + f, dataFormat=self.dataformats[f])
            self.assertEqual(wc.w_set(), True)
            self.assertEqual(wc.w.n, self.ns[f])

    def test_write(self):
        for f in self.test_files:
            with warnings.catch_warnings(record=True) as warn:
                # note: we are just suppressing the warnings here; individual warnings 
                #       are tested in their specific readers
                warnings.simplefilter("always")
                wc = WeightConverter(self.base_dir + f, dataFormat=self.dataformats[f])

            for ext, dataformat in self.fileformats:
                if f.lower().endswith(ext):
                    continue
                temp_f = tempfile.NamedTemporaryFile(suffix='.%s' % ext,dir=self.base_dir)
                temp_fname = temp_f.name
                temp_f.close()
               
                with warnings.catch_warnings(record=True) as warn:
                    # note: we are just suppressing the warnings here; individual warnings 
                    #       are tested in their specific readers
                    warnings.simplefilter("always")
                    if ext == 'swm':
                        wc.write(temp_fname,useIdIndex=True) 
                    elif dataformat is None:
                        wc.write(temp_fname)
                    elif dataformat in ['arcgis_dbf','arcgis_text']:
                        wc.write(temp_fname, dataFormat=dataformat, useIdIndex=True)
                    elif dataformat == 'stata_text':
                        wc.write(temp_fname, dataFormat=dataformat, matrix_form=True)
                    elif dataformat is None:
                        wc.write(temp_fname)
                    else:
                        wc.write(temp_fname, dataFormat=dataformat)

                with warnings.catch_warnings(record=True) as warn:
                    # note: we are just suppressing the warnings here; individual warnings 
                    #       are tested in their specific readers
                    warnings.simplefilter("always")
                    if dataformat is None:
                        wnew =  pysal.open(temp_fname,'r').read()
                    else:
                        wnew =  pysal.open(temp_fname,'r', dataformat).read()

                if (ext in ['dbf', 'swm', 'dat', 'wk1'] or dataformat == 'arcgis_text'):
                    self.assertEqual(wnew.n, wc.w.n - len(wc.w.islands))
                else:
                    self.assertEqual(wnew.n, wc.w.n)
                os.remove(temp_fname)

if __name__ == '__main__':
    unittest.main()
