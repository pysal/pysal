import unittest
import pysal.examples as ex


class Example_Tester(unittest.TestCase):
    def test_get_path(self):
        pathparts = ex.get_path('').split('/')
        self.localpath = '/'.join(pathparts[-3:])
        self.assertEquals(self.localpath, 'pysal/examples/')

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
