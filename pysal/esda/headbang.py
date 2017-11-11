# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import pysal
from ..weights import comb, Kernel, W
from ..weights.util import get_points_array
from ..cg import Point, Ray, LineSegment
from ..cg import get_angle_between, get_points_dist, get_segment_point_dist,\
                 get_point_at_angle_and_dist, convex_hull, get_bounding_box
from ..common import np, KDTree, requires as _requires
from ..weights.spatial_lag import lag_spatial as slag
from scipy.stats import gamma, norm, chi2, poisson
from functools import reduce

from itertools import combinations
from six import iterkeys
import warnings


class Headbanging_Triples(object):
    @staticmethod
    def is_valid_triple(p0, p1, p2, angle):
        ray1 = Ray(p0, p1)
        ray2 = Ray(p0, p2)
        ang = abs(np.rad2deg(get_angle_between(ray1, ray2)))
        return ang > angle

    @staticmethod
    def construct_triples(p0, neighbors, angle):
        triple = []
        for i1, i2 in combinations(iterkeys(neighbors), 2):
            if i1 > i2:  # Need to swap for consistency sake
                i2, i1 = i1, i2
            p1 = tuple(neighbors[i1])
            p2 = tuple(neighbors[i2])
            if Headbanging_Triples.is_valid_triple(p0, p1, p2, angle):
                triple.append(((p1, p2), (i1, i2)))
        return triple

    @staticmethod
    def construct_extra_triples(p0, neighbors, angle):
        extras = []
        for i1, i2 in combinations(iterkeys(neighbors), 2):
            p1 = tuple(neighbors[i1])
            p2 = tuple(neighbors[i2])
            extra = None
            if Headbanging_Triples.is_valid_triple(p1, p0, p2, 90 + angle / 2):
                extra = Headbanging_Triples.construct_one_extra(p0, p1, p2)
                ix1, ix2 = i1, i2
            elif Headbanging_Triples.is_valid_triple(p2, p0, p1,
                                                     90 + angle / 2):
                extra = Headbanging_Triples.construct_one_extra(p0, p2, p1)
                ix2, ix1 = i1, i2
            if extra:
                extras.append(((ix1, ix2),
                               get_points_dist(p1, p2),
                               get_points_dist(p0, extra)))
        extras = [(dist1, ix, dist1, dist2) for ix, dist1, dist2 in extras]
        if len(extras) > 0:
            extras = sorted(extras)[0]
            i1, i2, i3, i4 = extras
            return [i2, i3, i4]
        else:
            return []

    @staticmethod
    def construct_one_extra(p0, p1, p2):
        ray1 = Ray(p1, p0)
        ray2 = Ray(p1, p2)
        ang = get_angle_between(ray1, ray2)
        dist = get_points_dist(p0, p1)
        ray0 = Ray(p0, p1)
        return get_point_at_angle_and_dist(ray0, (2 * ang) - np.pi, dist)

    def __init__(self, data, w, t=3, angle=135.0, edgecor=False):
        if w.k < 3:
            raise ValueError("w should be NeareastNeighbors instance & the "
                             "number of neighbors should be more than 3.")
        if not w.id_order_set:
            raise ValueError("w id_order must be set to align with the order "
                             "of data")
        self.triples = {}
        for key in iterkeys(w.neighbors):
            p0 = tuple(data[key])
            neighbors_ix = w.neighbors[key]
            neighbor_points = data[neighbors_ix]
            neighbors = {
                    ix: tuple(neighbor_points[i])
                    for i, ix in enumerate(neighbors_ix)
                    }
            triples = Headbanging_Triples.construct_triples(p0, neighbors,
                                                            angle)
            if len(triples) > 3:
                triple_dis = []
                for points, triple in triples:
                    dist = get_segment_point_dist(
                            LineSegment(points[0], points[1]), p0)
                    triple_dis.append((dist, triple))
                triples = triple_dis[:t]
            if not edgecor and len(triples) == 0:
                warnings.warn('Index %s has no eligible triple and edge-'
                              'correction is off. Consider adding more '
                              'neighbors or using a smaller angle.' % key)
            self.triples[key] = [triple for _, triple in triples]
        if edgecor:
            self.extra = {}
            for key in iterkeys(self.triples):
                if len(self.triples[key]) == 0:
                    p0 = tuple(data[key])
                    neighbors_ix = w.neighbors[key]
                    neighbor_points = data[neighbors_ix]
                    neighbors = {
                            ix: tuple(neighbor_points[i])
                            for i, ix in enumerate(neighbors_ix)
                            }
                    extra = Headbanging_Triples.construct_extra_triples(
                            p0,
                            neighbors,
                            angle
                            )
                    if extra == []:
                        warnings.warn('edge-correction failed for index %s. '
                                      'Consider adding more neighbors or '
                                      'using a smaller angle.' % key)
                    else:
                        self.extra[key] = extra
