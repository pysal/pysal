#!/usr/bin/python
"""Sets and unsets a lock on the local pysal repository."""
import os, sys

#check lock
if os.path.exists('/tmp/pysal.lock'):
    print "LOCK IN PLACE, another process is running perhaps?"
    sys.exit(1)
else:
    lck = open('/tmp/pysal.lock','w')
    lck.write('%d'%os.getpid())
    lck.close()
    lck = True
    os.system('/Users/stephens/Dropbox/work/Projects/pysal/trunk/tools/test.sh')
    os.remove('/tmp/pysal.lock')

