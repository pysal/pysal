"""
PySAL Unit Testing

Prior to commiting any changes to the trunk, said changes should be checked
against the rest of the local copy of the trunk by running::

    python tests.py

If all tests pass, changes have not broken the current trunk and can be
committed. Commits that introduce breakage should only be done in cases where
other developers are notified and the breakage raises important issues for
discussion.


Notes
-----
Unit tests should be added for all modules in pysal.

To deal with relative paths in the doctests a symlink must first be made from
within the `tests` directory as follows::

     ln -s ../examples .

"""

__author__ = "Sergio J. Rey <srey@asu.edu>"

import unittest

suite = unittest.TestSuite()
# Test imports
try:
    import rtree
    import test_cont_weights
    suite.addTest(test_cont_weights.suite)
except ImportError:
    print "Cannot test rtree contiguity weights, rtree not installed"
import test_fileIO
suite.addTest(test_fileIO.suite)
import test_cg_shapes
suite.addTest(test_cg_shapes.suite)
import test_smoothing
suite.addTest(test_smoothing.suite)
import test_dist_weights
suite.addTest(test_dist_weights.suite)
import test_weights
suite.addTest(test_weights.suite)

runner = unittest.TextTestRunner()
runner.run(suite)
