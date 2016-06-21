from .. import stats
from ....pdio import read_files as rf
import unittest as ut

class ESDA_Mixin(object):
    def __init__(self, func=None, **facts):
        self.testing = func
        raise

    def test_columnar(self):
        raise

    def test_metadata_scrape(self):
        raise
    
    def test_inplace(self):
        raise

    def test_attribute_export(self):
        raise

class Test_Gamma(ut.TestCase, ESDA_Mixin):
    raise
class Test_Geary(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran_Local(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran_BV(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran_Local_BV(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran_Rate(ut.TestCase, ESDA_Mixin):
    raise
class Test_Moran_Local_Rate(ut.TestCase, ESDA_Mixin):
    raise
