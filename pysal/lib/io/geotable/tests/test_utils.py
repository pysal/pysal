from .. import utils
from ..file import read_files as rf
import unittest as ut

@ut.skip('skpping converters and metadata inserters')
class Test_Utils(ut.TestCase):
    
    def test_converters(self):
        ## make a round trip to geodataframe and back
        raise Exception
    def test_insert_metadata(self):
        ## add an attribute to a dataframe and see 
        ## if it is pervasive over copies
        raise Exception
