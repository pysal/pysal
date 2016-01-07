#!/usr/bin/env python2 

import nose_json
import nosetimer
import memorygrind
import nose
import shutil as sh
import os
import subprocess

print((os.getcwd()))
print(__file__)

currwarn = os.environ.get('PYTHONWARNINGS')
os.environ['PYTHONWARNINGS'] = 'ignore'

subprocess.call("nosetests-2.7 --collect-only --with-tissue &> pysal/contrib/profiling/tissue.txt", shell=True)
subprocess.call("nosetests-2.7 --with-nose-memory-grind --verbose --with-json --json-file=pysal/contrib/profiling/nosetests.json --with-timer &> pysal/contrib/profiling/memtest.txt", shell=True)

if currwarn is None:
    currwarn = 'default'
os.environ['PYTHONWARNINGS'] = currwarn
