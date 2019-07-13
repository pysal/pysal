from pysal.viz.splot._viz_utils import shift_colormap, truncate_colormap
import matplotlib as mpl


def test_shift_colormap():
    map_test = shift_colormap('RdBu', start=0.1,
                              midpoint=0.2,
                              stop=0.9,
                              name='shiftedcmap')
    assert isinstance(map_test, mpl.colors.LinearSegmentedColormap)


def test_truncat_colormap():
    map_test_truncate = truncate_colormap('RdBu', minval=0.1,
                                          maxval=0.9, n=99)
    assert isinstance(map_test_truncate, mpl.colors.LinearSegmentedColormap)
