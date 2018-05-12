from unittest import TestCase, skip
TestCase.maxDiff = None
from ...examples import available, explain

@skip
class TestHelpers(TestCase):
    def test_available(self):
        examples = available()
        self.assertEqual(len(examples), 31)
        self.assertIn('arcgis', examples)
        self.assertIn('nat', examples)

    def test_explain(self):
        des = 'Homicides and selected socio-economic characteristics for counties surrounding St Louis, MO. Data aggregated for three time periods: 1979-84 (steady decline in homicides), 1984-88 (stable period), and 1988-93 (steady increase in homicides).'
        e = explain('stl')
        self.assertEqual(e['description'], des)
        self.assertEqual(len(e['explanation']), 10)


if __name__ == '__main__':
    import unittest
    unittest.main()
