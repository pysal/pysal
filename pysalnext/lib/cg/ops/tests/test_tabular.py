import unittest as ut
from .. import tabular as ta
from ....common import RTOL, ATOL, pandas, requires as _requires
from ....examples import get_path
from ...shapes import Polygon
from ....io import geotable as pdio
from ... import ops as GIS
import numpy as np

try:
    import shapely as shp
except ImportError:
    shp = None

PANDAS_EXTINCT = pandas is None
SHAPELY_EXTINCT = shp is None

@ut.skipIf(PANDAS_EXTINCT or SHAPELY_EXTINCT, 'missing pandas or shapely')
class Test_Tabular(ut.TestCase):
    def setUp(self):
        import pandas as pd
        self.columbus = pdio.read_files(get_path('columbus.shp'))
        grid = [Polygon([(0,0),(0,1),(1,1),(1,0)]),
                Polygon([(0,1),(0,2),(1,2),(1,1)]),
                Polygon([(1,2),(2,2),(2,1),(1,1)]),
                Polygon([(1,1),(2,1),(2,0),(1,0)])]
        regime = [0,0,1,1]
        ids = list(range(4))
        data = np.array((regime, ids)).T
        self.exdf = pd.DataFrame(data, columns=['regime', 'ids'])
        self.exdf['geometry'] = grid

    @_requires('geopandas')
    def test_round_trip(self):
        import geopandas as gpd
        import pandas as pd
        geodf = GIS.tabular.to_gdf(self.columbus)
        self.assertIsInstance(geodf, gpd.GeoDataFrame)
        new_df = GIS.tabular.to_df(geodf)
        self.assertIsInstance(new_df, pd.DataFrame)
        for new, old in zip(new_df.geometry, self.columbus.geometry):
            self.assertEqual(new, old)

    def test_spatial_join(self):
        pass

    def test_spatial_overlay(self):
        pass
    
    def test_dissolve(self):
        out = GIS.tabular.dissolve(self.exdf, by='regime')
        self.assertEqual(out[0].area, 2.0)
        self.assertEqual(out[1].area, 2.0)

        answer_vertices0 = [(0,0), (0,1), (0,2), (1,2), (1,1), (1,0), (0,0)]
        answer_vertices1 = [(2,1), (2,0), (1,0), (1,1), (1,2), (2,2), (2,1)]

        np.testing.assert_allclose(out[0].vertices, answer_vertices0)
        np.testing.assert_allclose(out[1].vertices, answer_vertices1)

    def test_clip(self):
        pass

    def test_erase(self):
        pass

    def test_union(self):
       new_geom =  GIS.tabular.union(self.exdf)
       self.assertEqual(new_geom.area, 4)

    def test_intersection(self):
        pass

    def test_symmetric_difference(self):
        pass

    def test_difference(self):
        pass
