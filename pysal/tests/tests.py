"""
Integration testing for PySAL

Each package in PySAL shall have a tests directory. Within this directory shall
be one test_module.py file for each module.py in the package.

Note
----

End of this file will eventually be depreciated and removed as it was the 1.0
testing scheme which will be replaced with this file.

"""

import os
import unittest

path="../"
skip=[".svn","tests"]


# test for existence of test_*.py in mod/tests

runners=[]
missing=[]

for root,subfolders,files in os.walk(path):
    for ignore in skip:
        if ignore in subfolders:
            subfolders.remove(ignore)
    mods=[fname for fname in files if fname.endswith(".py")]
    tests=[os.path.join(root,"tests","test_"+mod) for mod in mods]
    for test in tests:
        if os.path.exists(test):
            runners.append(test)
        else:
            missing.append(test)

import time
cwd=os.path.abspath(".")
t1=time.time()
for _test in runners:
    print "Unit testing: ",_test
    pth,file=os.path.split(_test)
    apth=os.path.abspath(pth)
    print pth,file,apth
    os.chdir(apth)
    execfile(file)
    os.chdir(cwd)
t2=time.time()
print t2-t1
os.chdir(cwd)

print "Untested methods"
for missed in missing:
    print missed


print "Running old_tests"

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
import test_weights
suite.addTest(test_weights.suite)

runner = unittest.TextTestRunner()
runner.run(suite)
