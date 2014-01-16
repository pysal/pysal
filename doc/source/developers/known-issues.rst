
Known Issues
============

1.5 install fails with scipy 11.0 on Mac OS X
---------------------------------------------

Running `python setup.py install` results in::

	from _cephes import *
	ImportError:
	dlopen(/Users/serge/Documents/p/pysal/virtualenvs/python1.5/lib/python2.7/site-packages/scipy/special/_cephes.so,
	2): Symbol not found: _aswfa_
	  Referenced from:
	  /Users/serge/Documents/p/pysal/virtualenvs/python1.5/lib/python2.7/site-packages/scipy/special/_cephes.so
	    Expected in: dynamic lookup

This occurs when your scipy on Mac OS X was complied with gnu95 and not
gfortran.  See `this thread <http://mail.scipy.org/pipermail/scipy-user/2010-November/027548.html>`_ for possible solutions.

weights.DistanceBand failing
----------------------------

This occurs due to a bug in scipy.sparse prior to version 0.8. If you are running such a version see `Issue 73 <http://code.google.com/p/pysal/issues/detail?id=73&sort=milestone>`_ for a fix.

doc tests and unit tests under Linux
------------------------------------

Some Linux machines return different results for the unit and doc tests. We suspect this has to do with the way random seeds are set. See `Issue 52 <http://code.google.com/p/pysal/issues/detail?id=52&sort=milestone>`_

LISA Markov missing a transpose
-------------------------------
In versions of PySAL < 1.1 there is a bug in the LISA Markov, resulting in
incorrect values. For a fix and more details see `Issue 115 <http://code.google.com/p/pysal/issues/detail?id=115>`_.


PIP Install Fails
-----------------


Having numpy and scipy specified in pip requiretments.txt causes PIP install of pysal to fail. For discussion and suggested fixes see `Issue 207 <http://code.google.com/p/pysal/issues/detail?id=207&sort=milestone>`_.

