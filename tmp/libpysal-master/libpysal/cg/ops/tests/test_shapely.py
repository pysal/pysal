import unittest as ut
from .. import _shapely as sht
from ...shapes import Point, Chain, Polygon
#from ... import  comparators as comp
#from ... import shapely as she
from ....io.geotable import read_files as rf
from ....examples import get_path
import numpy as np
from warnings import warn

@ut.skip('skipping shapely during reorg')
class Test_Shapely(ut.TestCase):
    def setUp(self):
        self.polygons = rf(get_path('Polygon.shp'))
        self.points = rf(get_path('Point.shp'))
        self.lines = rf(get_path('Line.shp'))
        self.target_poly = self.polygons.geometry[2]
        self.target_point = self.points.geometry[1]
        self.target_line = self.lines.geometry[0]

        self.dframes = [self.polygons, self.points, self.lines]
        self.targets = [self.target_poly, self.target_point, self.target_line]

    def compare(self, func_name, df, **kwargs):
        geom_list = df.geometry.tolist()
        shefunc = she.__dict__[func_name]
        shtfunc = sht.__dict__[func_name]

        try:
            she_vals = (shefunc(geom, **kwargs) for geom in geom_list)
            sht_vals = shtfunc(df, inplace=False, **kwargs)
            sht_list = sht_vals['shape_{}'.format(func_name)].tolist()
            for tabular, shapely in zip(sht_list, she_vals):
                if (comp.is_shape(tabular) and
                    comp.is_shape(shapely)):
                    comp.equal(tabular, shapely)
                else:
                    self.assertEqual(tabular, shapely)
        except NotImplementedError as e:
            warn('The shapely/pysal bridge is not implemented: {}'.format(e))
            return True

    def test_to_wkb(self):
        for df in self.dframes:
            self.compare('to_wkb', df)

    def test_to_wkt(self):
        for df in self.dframes:
            self.compare('to_wkt', df)

    def test_area(self):
        for df in self.dframes:
            self.compare('area', df)

    def test_distance(self):
        for df in self.dframes:
            for other in self.targets:
                self.compare('distance', df, other=other)

    def test_length(self):
        for df in self.dframes:
            self.compare('length', df)

    def test_boundary(self):
        for df in self.dframes:
            self.compare('boundary', df)

    def test_bounds(self):
        for df in self.dframes:
            self.compare('bounds', df)

    def test_centroid(self):
        for df in self.dframes:
            self.compare('centroid', df)

    def test_representative_point(self):
        for df in self.dframes:
            self.compare('representative_point', df)

    def test_convex_hull(self):
        for df in self.dframes:
            self.compare('convex_hull', df)

    def test_envelope(self):
        for df in self.dframes:
            self.compare('envelope', df)

    def test_buffer(self):
        np.random.seed(555)
        for df in self.dframes:
            self.compare('buffer', df, radius=np.random.randint(10))

    def test_simplify(self):
        tol = .001
        for df in self.dframes:
            self.compare('simplify', df, tolerance=tol)

    def test_difference(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('difference', df, other=target)

    def test_intersection(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('intersection', df, other=target)

    def test_symmetric_difference(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('symmetric_difference', df, other=target)

    def test_union(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('union', df, other=target)

    def test_has_z(self):
        for df in self.dframes:
            self.compare('has_z', df)

    def test_is_empty(self):
        """
        PySAL doesn't really support empty shapes. Like, the following errors
        out:

        ps.cg.Polygon([[]])

        and you can make it work by:
        ps.cg.Polygon([[()]])

        but that won't convert over to shapely.

        So, we're only testing the negative here.
        """
        for df in self.dframes:
            self.compare('is_empty', df)

    def test_is_ring(self):
        for df in self.dframes:
            self.compare('is_ring', df)

    def test_is_simple(self):
        for df in self.dframes:
            self.compare('is_simple', df)

    def test_is_valid(self):
        for df in self.dframes:
            self.compare('is_valid', df)

    def test_relate(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('relate', df, other=target)

    def test_contains(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('contains', df, other=target)

    def test_crosses(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('crosses', df, other=target)

    def test_disjoint(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('disjoint', df, other=target)

    def test_equals(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('equals', df, other=target)

    def test_intersects(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('intersects', df, other=target)

    def test_overlaps(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('overlaps', df, other=target)

    def test_touches(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('touches', df, other=target)

    def test_within(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('within', df, other=target)

    def test_equals_exact(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('equals_exact', df, other=target, tolerance=.1)

    def test_almost_equals(self):
        for df in self.dframes:
            for target in self.targets:
                self.compare('almost_equals', df, other=target)

    def test_project(self):
        np.random.seed(555)
        self.compare('project', self.lines, other=self.targets[2])

    def test_interpolate(self):
        np.random.seed(555)
        for df in self.dframes:
            if isinstance(df.geometry[0], Chain):
                self.compare('interpolate', df, distance=np.random.randint(10))
            else:
                with self.assertRaises(TypeError):
                    self.compare('interpolate', df, distance=np.random.randint(10))
