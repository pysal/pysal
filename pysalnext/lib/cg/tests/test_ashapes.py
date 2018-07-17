from unittest import TestCase, skipIf
from ...examples import get_path
from ..alpha_shapes import alpha_shape, alpha_shape_auto
import numpy as np
import os

try:
    import geopandas
    GEOPANDAS_EXTINCT = False
except ImportError:
    GEOPANDAS_EXTINCT = True

this_directory = os.path.dirname(__file__)


@skipIf(GEOPANDAS_EXTINCT, 'Geopandas is missing, so test will not run')
class Test_Alpha_Shapes(TestCase):
    def setUp(self):
        eberly = geopandas.read_file(get_path('eberly_net.shp'))
        eberly_vertices = eberly.geometry.apply(lambda x: np.hstack(x.xy).reshape(2, 2).T).values
        eberly_vertices = np.vstack(eberly_vertices)
        self.vertices = eberly_vertices

        self.a05 = geopandas.read_file(os.path.join(this_directory, 'data/alpha_05.shp')).geometry.item()
        self.a10 = geopandas.read_file(os.path.join(this_directory, 'data/alpha_tenth.shp')).geometry.item()
        self.a2 = geopandas.read_file(os.path.join(this_directory, 'data/alpha_fifth.shp')).geometry.item()
        self.a25 = geopandas.read_file(os.path.join(this_directory, 'data/alpha_fourth.shp')).geometry.item()
        self.a25 = geopandas.read_file(os.path.join(this_directory, 'data/alpha_fourth.shp')).geometry.item()

        self.autoalpha = geopandas.read_file(os.path.join(this_directory, 'data/alpha_auto.shp')).geometry.item()

    def test_alpha_shapes(self):
        new_a05 = alpha_shape(self.vertices, .05).item()
        new_a10 = alpha_shape(self.vertices, .10).item()
        new_a2 = alpha_shape(self.vertices, .2).item()
        new_a25 = alpha_shape(self.vertices, .25).item()

        assert new_a05.equals(self.a05)
        assert new_a10.equals(self.a10)
        assert new_a2.equals(self.a2)
        assert new_a25.equals(self.a25)

    def test_auto(self):
        auto_alpha = alpha_shape_auto(self.vertices, 5)

        assert self.autoalpha.equals(auto_alpha)
