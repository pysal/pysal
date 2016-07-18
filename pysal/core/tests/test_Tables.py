import pysal as ps
import numpy as np
import unittest as ut

PANDAS_EXTINCT = ps.common.pandas is None

class Test_Table(ut.TestCase):
    def setUp(self):
        self.filehandler = ps.open(ps.examples.get_path('columbus.dbf'))
        self.df = self.filehandler.to_df()
        self.filehandler.seek(0)
        self.shapefile = ps.open(ps.examples.get_path('columbus.shp'))
        self.csvhandler = ps.open(ps.examples.get_path('usjoin.csv'))
        self.csv_df = self.csvhandler.to_df()
        self.csvhandler.seek(0)
    
    @ut.skipIf(PANDAS_EXTINCT, 'missing pandas')
    def test_to_df(self):
        for column in self.csv_df.columns:
            if column.lower() == 'name':
                continue
            np.testing.assert_allclose(self.csvhandler.by_col(column), 
                                       self.csv_df[column].values)
        for column in self.df.columns:
            if column == 'geometry':
                continue
            np.testing.assert_allclose(self.filehandler.by_col(column), 
                                       self.df[column])

