import unittest
import pysal.lib
import geopandas as gpd
from pysal.explore.segregation.network import get_osm_network, calc_access


@unittest.skip("Skipping Network_Tester")
class Network_Tester(unittest.TestCase):
    def test_calc_access(self):
        variables = ["WHITE_", "BLACK_", "ASIAN_", "HISP_"]
        s_map = gpd.read_file(pysal.lib.examples.get_path("sacramentot2.shp"))
        df = s_map[["FIPS", "geometry"] + variables]
        df = df[df.FIPS.str.startswith("06061")]
        df = df[(df.centroid.x < -121) & (df.centroid.y < 38.85)]
        df.crs = {"init": "epsg:4326"}
        df[variables] = df[variables].astype(float)
        test_net = get_osm_network(df, maxdist=0)
        acc = calc_access(df, test_net, distance=1.0, variables=variables)
        assert acc.acc_WHITE_.sum() > 100


if __name__ == "__main__":
    unittest.main()
