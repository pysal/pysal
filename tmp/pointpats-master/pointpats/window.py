"""
Window class for point patterns
"""

__author__ = "Serge Rey sjsrey@gmail.com"

import libpysal as ps
import numpy as np
__all__ = ["as_window", "poly_from_bbox", "to_ccf", "Window"]


def poly_from_bbox(bbox):
    l, b, r, t = bbox
    c = [(l, b), (l, t), (r, t), (r, b), (l, b)]
    return ps.cg.shapes.Polygon(c)


def to_ccf(poly):
    if poly[-1] != poly[0]:
        poly.append(poly[0])
    return poly


def as_window(pysal_polygon):
    """
    Convert a libpysal polygon to a Window.

    Parameters
    ----------
    pysal_polygon: libpysal.cg.shapes.Polygon
                   libpysal Polygon instance.

    Returns
    -------
    Window
                   A Window instance.
    """

    if pysal_polygon.holes == [[]]:
        return Window(pysal_polygon.parts)
    else:
        return Window(pysal_polygon.parts, pysal_polygon.holes)


class Window(ps.cg.Polygon):
    """
    Geometric container for point patterns.

    A window is used to define the area over which the pattern is observed.
    This area is used in estimating the intensity of the point pattern.
    See :attr:`PointPattern.lambda_window`.

    Parameters
    ----------
    parts: sequence
           A sequence of rings which bound the positive space  point
           pattern.
    holes: sequence
           A sequence of rings which bound holes in the polygons that bound the
           point pattern.

    """
    def __init__(self, parts, holes=[]):

        if holes:
            super(Window, self).__init__(parts, holes)
        else:
            super(Window, self).__init__(parts)

    def filter_contained(self, points):
        return [np.asarray(pnt) for pnt in points if self.contains_point(pnt)]
