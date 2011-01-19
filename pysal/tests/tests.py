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

path = "../"
skip = [".svn", "tests"]


# test for existence of test_*.py in mod/tests
not_testing = ['common.py', 'version.py', '__init__.py' ]
runners = []
missing = []
missing_all = []
expectedUnits = {}
missingUnits = {}

for root, subfolders, files in os.walk(path):
    for ignore in skip:
        if ignore in subfolders:
            subfolders.remove(ignore)
    #mods = [fname for fname in files if fname.endswith(".py")]
    mods = [fname for fname in files if fname.endswith(".py") and \
                                        fname not in not_testing ]
    tests = [os.path.join(root, "tests", "test_"+mod) for mod in mods]
    for mod, testMod in zip(mods, tests):
        mod = os.path.join(root, mod)
        if testMod not in expectedUnits:
            expectedUnits[testMod] = []
        if "__all__" not in open(mod, 'r').read():
            missing_all.append(mod)
        else:
            lines = [l for l in open(mod, 'r') if "__all__" in l]
            if len(lines) > 1 or ']' not in lines[0]:
                print "Ambiguous __all__ in", mod
            else:
                l = lines[0].split('[')[1].strip().replace(']','').replace('"','').replace("'","").replace(' ','')
                for x in l.split(','):
                    expectedUnits[testMod].append("test_"+x)
    for test in tests:
        if os.path.exists(test):
            txt = open(test, 'r').read()
            if txt:
                runners.append(test)
                missingUnits[test] = []
                for unit in expectedUnits[test]:
                    if unit not in txt:
                        missingUnits[test].append(unit)
            else:
                missing.append(test)
        else:
            missing.append(test)
        
#import time
cwd = os.path.abspath(".")
#t1 = time.time()
for _test in runners:
    print "Unit testing: ", _test
    pth, fname = os.path.split(_test)
    apth = os.path.abspath(pth)
    #print pth, fname, apth
    os.chdir(apth)
    execfile(fname)
    os.chdir(cwd)
#t2 = time.time()
#print t2-t1
os.chdir(cwd)

print "----------------------------------------"
print ''
print "Modules Missing __all__:"
print ''
for missed in missing_all:
    print "__all__ is not defined in", missed
print ''
print "----------------------------------------"
print ''
print "Missing Test Modules:"
print ''
for missed in missing:
    print missed
print ''
print "----------------------------------------"
print ''
print "Untested Methods in Tested Modules:"
print ''
for key in missingUnits:
    if missingUnits[key]:
        print key, " is missing expected test(s): ",','.join(missingUnits[key])
print ''
print "----------------------------------------"
print ''
print "Running doc_tests"

__author__ = "Sergio J. Rey <srey@asu.edu>"

import unittest

suite = unittest.TestSuite()
# Test imports
#import test_fileIO
#suite.addTest(test_fileIO.suite)
#import test_cg_shapes
#suite.addTest(test_cg_shapes.suite)
import test_weights
import test_NameSpace
suite.addTest(test_weights.suite)
suite.addTest(test_NameSpace.suite)

runner = unittest.TextTestRunner()
runner.run(suite)

