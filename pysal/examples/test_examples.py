import unittest
import pysal.examples as ex
import os


class Example_Tester(unittest.TestCase):
    def test_get_path(self):
        pathparts = os.path.normpath(ex.get_path('')).split(os.path.sep)
        self.localpath = os.path.join(*pathparts[-2:])
        self.assertEquals(self.localpath, os.path.normpath('pysal/examples/'))

    def test_parser(self):
        for example in ex.available():
            self.extext = ex.explain(example)
            # make sure names are returned
            self.assertIsNotNone(self.extext['name'])
            # Make sure there's an empty line between titling and description
            if self.extext['description'] is not None:
                self.assertNotIn('----', self.extext['description'])

if __name__ == '__main__':
    unittest.main()
