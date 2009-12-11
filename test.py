import unittest
import doctest

### Find Modules
import os,types
def get_modules(module,mods=set()):
    PATH = os.path.dirname(module.__file__)
    a = [getattr(module,mod) for mod in dir(module)]
    b = [mod for mod in a if type(mod)==types.ModuleType]
    c = [mod for mod in b if mod.__file__.startswith(PATH)]
    d = [mod for mod in c if mod not in mods]
    mods.update(set(d[:]))
    print module,len(mods)
    for mod in d:
        mods.update(get_modules(mod,mods))
    return mods

import pysal
suite = unittest.TestSuite()
for mod in get_modules(pysal):
    try:
        suite.addTest(doctest.DocTestSuite(mod))
    except ValueError:
        print "No Tests in module: ",mod
runner = unittest.TextTestRunner()
runner.run(suite)
