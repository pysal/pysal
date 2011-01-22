"""
Package-wide doc tests for PySAL

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
import pysal.esda.moran, pysal.esda.geary, pysal.esda.join_counts
import pysal.esda.mapclassify
import pysal.inequality.theil
import pysal.region.maxp
import pysal.region.randomregion
import pysal.spatial_dynamics.rank
import pysal.spatial_dynamics.markov
import pysal.spatial_dynamics.ergodic
import pysal.spatial_dynamics.directional
import pysal.weights.spatial_lag, pysal.weights.util
import pysal.weights.Contiguity, pysal.weights.Distance, pysal.weights.user
import pysal.weights.Wsets
import pysal.esda.smoothing
import pysal.cg.locators

#add modules to include in tests
mods='esda.moran','esda.geary', 'esda.mapclassify', \
        'esda.join_counts', \
        'spatial_dynamics.rank', \
        'inequality.theil', \
        'region.maxp', 'region.randomregion', \
        'weights', \
        'weights.spatial_lag', 'weights.util', \
        'weights.Contiguity', 'weights.Distance', 'weights.user', \
        'weights.Wsets', 'esda.smoothing', \
        'spatial_dynamics.markov', \
        'spatial_dynamics.ergodic' , \
        'spatial_dynamics.directional', \
        'cg.locators', \
        'spreg.ols', 'spreg.diagnostics', 'spreg.diagnostics_sp', \
        'spreg.user_output' 

mods = [ "pysal."+ mod for mod in mods]
suite = unittest.TestSuite()
for mod in mods:
    suite.addTest(doctest.DocTestSuite(mod))

runner = unittest.TextTestRunner()
runner.run(suite)
