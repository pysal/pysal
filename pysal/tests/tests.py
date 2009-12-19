"""
Package-wide tests for PySAL

Currently allows for testing of docstring examples from individual modules.
Prior to commiting any changes to the trunk, said changes should be checked
against the rest of the local copy of the trunk by running::

    python tests.py

If all tests pass, changes have not broken the current trunk and can be
committed. Commits that introduce breakage should only be done in cases where
other developers are notified and the breakage raises important issues for
discussion.


Notes
-----
New modules need to be included in the `#module imports` section below, as
well as in the truncated module list where `mods` is first defined.

To deal with relative paths in the doctests a symlink must first be made from
within the `tests` directory as follows::

     ln -s ../examples .

"""

__author__ = "Sergio J. Rey <srey@asu.edu>"

import unittest
import doctest

# module imports
import pysal.esda.moran, pysal.esda.geary
import pysal.esda.mapclassify
import pysal.inequality.theil
import pysal.econometrics.classic
import pysal.mobility.rank
import pysal.weights.weights, pysal.weights.ContiguityWeights
import pysal.weights.DistanceWeights, pysal.weights.spatial_lag


#add modules to include in tests
mods='esda.moran','esda.geary', 'esda.mapclassify', \
        'econometrics.classic', \
        'inequality.theil','mobility.rank', \
        'weights.weights','weights.DistanceWeights', \
        'weights.ContiguityWeights','weights.spatial_lag'
mods = [ "pysal."+ mod for mod in mods]
suite = unittest.TestSuite()
for mod in mods:
    suite.addTest(doctest.DocTestSuite(mod))
runner = unittest.TextTestRunner()
runner.run(suite)
