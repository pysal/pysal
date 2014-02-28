#############################
Introduction to the Tutorials
#############################


Assumptions
===========

The tutorials presented here are designed to illustrate a selection of the
functionality in PySAL. Further details on PySAL functionality not covered in
these tutorials can be found in the :doc:`API <../../library/index>`. The reader is
**assumed to have working knowledge of the particular spatial analytical
methods** illustrated. Background on spatial analysis can be found in the
references cited in the tutorials.

It is also assumed that the reader has already :doc:`installed PySAL <../installation>`.

Examples
============

The examples use several sample data sets that are included in the pysal/examples
directory. In the examples that follow, we refer to those using the path::

  ../pysal/examples/filename_of_example
 
You may need to adjust this path to match the location of the sample files on
your system.


Getting Help
============

Help for PySAL is available from a number of sources.

email lists
-----------
The main channel for user support is the `openspace mailing list <http://groups.google.com/group/openspace-list>`_.


Questions regarding the development of PySAL should be directed to
`pysal-dev  <http://groups.google.com/group/pysal-dev>`_.

Documentation
-------------
Documentation is available on-line at `pysal.org <http://pysal.org>`_.

You can also obtain help at the interpreter:

	>>> import pysal
	>>> help(pysal)
	
which would bring up help on PySAL::

	Help on package pysal:

	NAME
	    pysal

	FILE
	    /Users/serge/Dropbox/pysal/src/trunk/pysal/__init__.py

	DESCRIPTION
	    Python Spatial Analysis Library
	    ===============================
	    
	    
	    Documentation
	    -------------
	    PySAL documentation is available in two forms: python docstrings and a html webpage at http://pysal.org/
	    
	    Available sub-packages
	    ----------------------
	    
	    cg
	:

Note that you can use this on any option within PySAL:

	>>> w=pysal.lat2W()
	>>> help(w)

which brings up::

	Help on W in module pysal.weights object:

	class W(__builtin__.object)
	 |  Spatial weights
	 |  
	 |  Parameters
	 |  ----------
	 |  neighbors       : dictionary
	 |                    key is region ID, value is a list of neighbor IDS
	 |                    Example:  {'a':['b'],'b':['a','c'],'c':['b']}
	 |  weights = None  : dictionary
	 |                    key is region ID, value is a list of edge weights
	 |                    If not supplied all edge wegiths are assumed to have a weight of 1.
	 |                    Example: {'a':[0.5],'b':[0.5,1.5],'c':[1.5]}
	 |  id_order = None : list 
	 |                    An ordered list of ids, defines the order of
	 |                    observations when iterating over W if not set,
	 |                    lexicographical ordering is used to iterate and the
	 |                    id_order_set property will return False.  This can be
	 |                    set after creation by setting the 'id_order' property.
	 |  
	 

Note that the help is truncated at the bottom of the terminal window and more of the contents can be seen by scrolling (hit any key).
	
