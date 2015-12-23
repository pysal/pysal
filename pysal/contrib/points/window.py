"""
Window class for point patterns
"""

__author__ = "Serge Rey sjsrey@gmail.com"

import pysal as ps
import numpy as np


def poly_from_bbox(bbox):
    l, b, r, t = bbox
    c = [(l, b), (l, t), (r, t), (r, b), (l, b)]
    return ps.cg.shapes.Polygon(c)


def as_window(pysal_polygon):
    if pysal_polygon.holes == [[]]:
        return Window(pysal_polygon.parts)
    else:
        return Window(pysal_polygon.parts, pysal_polygon.holes)


class Window(ps.cg.shapes.Polygon):
    def __init__(self, parts, holes=[]):
        if holes:
            super(Window, self).__init__(parts, holes)
        else:
            super(Window, self).__init__(parts)

    def filter_contained(self, points):
        return [np.asarray(pnt) for pnt in points if self.contains_point(pnt)]
