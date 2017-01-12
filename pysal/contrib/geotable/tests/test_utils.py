from .. import utils
from ...pdio import read_files
import pysal as ps
import unittest as ut
try:
    import geopandas as gpd
except ImportError:
    gpd = None

MISSING_GEOPANDAS = gpd is None 

@ut.skipIf(MISSING_GEOPANDAS, 'geopandas is missing...')
class Test_Utils(ut.TestCase):
    def setUp(self):
        self.df = read_files(ps.examples.get_path('columbus.shp'))
        self.gdf = gpd.read_file(ps.examples.get_path('columbus.shp'))

    
    def test_converters(self):
        import shapely.geometry as shgeom
        import pandas as pd

        from_df = utils.to_gdf(self.df)
        from_gdf = utils.to_df(self.gdf)

        self.assertIsInstance(from_df, type(self.gdf))
        self.assertIsInstance(from_gdf, type(self.df))

        self.assertIsInstance(from_gdf.iloc[0].geometry,
                              ps.cg.Polygon)
        self.assertIsInstance(from_df.iloc[0].geometry,
                              shgeom.base.BaseGeometry)

        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertIsInstance(self.gdf,gpd.GeoDataFrame)

        self.assertIsInstance(self.df.iloc[0].geometry,
                              ps.cg.Polygon)
        self.assertIsInstance(self.gdf.iloc[0].geometry,
                              shgeom.base.BaseGeometry)

    def test_insert_metadata(self):
        ## add an attribute to a dataframe and see 
        ## if it is pervasive over copies
        W = ps.weights.Queen.from_dataframe(self.gdf)
        new = utils.insert_metadata(self.df, W, 'W', inplace=False)
        utils.insert_metadata(self.gdf, W, 'W', inplace=True)
        assert hasattr(new, 'W')
        self.assertIsInstance(new.W, ps.weights.W)

        assert hasattr(new.copy(), 'W') 

        assert hasattr(new.copy(deep=True), 'W')
        
        assert hasattr(self.gdf, 'W')
        
        try:
            utils.insert_metadata(self.gdf, W, 'W', 
                                  inplace=True, overwrite=False)
            raise UserWarning('Did not raise an exception when'
                              'overwriting and overwrite=False')
        except Exception:
            pass

if __name__ == "__main__":
    ut.main()
