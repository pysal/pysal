.. role:: strike

*******************
PySAL Documentation
*******************
.. contents::

.. _compiling-doc-label:


Writing Documentation
=====================

The PySAL project contains two distinct forms of documentation: inline and
non-inline. Inline docs are contained in the source
code itself, in what are known as *docstrings*.  Non-inline documentation is in the
doc folder in the trunk. 

Inline documentation is processed with an extension to Sphinx called napoleon.
We have adopted the community standard outlined `here`_.

PySAL makes use of the built-in Sphinx extension *viewcode*, which allows the
reader to quicky toggle between docs and source code. To use it,
the source code module requires at least one properly formatted docstring.

Non-inline documentation editors can opt to strike-through older documentation rather than
delete it with the custom "role" directive as
follows.  Near the top of the document, add the role directive.  Then, to strike through old text, add the :strike:
directive and offset the text with back-ticks. This :strike:`strikethrough` is produced
like this::

  .. role:: strike

  ...
  ...

  This :strike:`strikethrough` is produced like this:

Compiling Documentation
=======================
 
PySAL documentation is built using `Sphinx`_ and the Sphinx extension `napoleon`_, which formats PySAL's docstrings. 

Note
----
If you're using Sphinx version 1.3 or newer, napoleon is included and should be called in the main conf.py as sphinx.ext.napoleon rather than installing it as we show below.

If you're using a version of Sphinx that does not ship with napoleon ( Sphinx < 1.3), you'll need napoleon version 0.2.4 or later and Sphinx version 1.0 or later to compile the documentation. 
Both modules are available at the Python Package Index, and can be downloaded and installed
from the command line using *pip* or *easy_install*.::

       $ easy_install sphinx
       $ easy_install sphinxcontrib-napoleon
or

       $ pip sphinx
       $ pip sphinxcontrib-napoleon
              
If you get a permission error, trying using 'sudo'. 

The source for the docs is in `doc`. Building the documentation is
done as follows (assuming sphinx and napoleon are already installed)::

        $ cd doc; ls
        build  Makefile  source

        $ make clean
        $ make html

To see the results in a browser open `build/html/index.html`. To make
changes, edit (or add) the relevant files in `source` and rebuild the
docs using the 'make html' (or 'make clean' if you're adding new documents) command. 
Consult the `Sphinx markup guide`_ for details on the syntax and structure of the files in `source`.

Once you're happy with your changes, check-in the `source` files. Do not
add or check-in files under  `build` since they are dynamically built.

Changes checked in to `Github`_ will be propogated to `readthedocs`_ within a few minutes.


Lightweight Editing with rst2html.py
------------------------------------

Because the doc build process can sometimes be lengthy, you may want to avoid
having to do a full build until after you are done with your major edits on
one particular document.  As part of the
`docutils`_ package,
the file `rs2html.py` can take an `rst` document and generate the html file.
This will get most of the work done that you need to get a sense if your edits
are good, *without* having to rebuild all the PySAL docs. As of version 0.8 it
also understands LaTeX. It will cough on some sphinx directives, but those can
be dealt with in the final build.

To use this download the doctutils tarball and put `rst2html.py` somewhere in
your path. In vim (on Mac OS X) you can then add something like::

    map ;r ^[:!rst2html.py % > ~/tmp/tmp.html; open ~/tmp/tmp.html^M^M

which will render the html in your default browser.

Things to watch out for
------------------------

If you encounter a failing tutorial doctest that does not seem to be in error, it could be 
a difference in whitespace between the expected and received output. In that case, add an 
'options' line as follows::
 
 .. doctest::
    :options: +NORMALIZE_WHITESPACE
	
    >>> print 'a   b   c'
    abc

Adding a new package and modules
================================

To include the docstrings of a new module in the :doc:`API docs </library/index>` the following steps are required:

 1. In the directory `/doc/source/library` add a directory with the name of
    the new package. You can skip to step 3 if the package exists and you are
    just adding new modules to this package.
 2. Within `/doc/source/library/packageName` add a file `index.rst`
 3. For each new module in this package, add a file `moduleName.rst` and
    update the `index.rst` file to include `modulename`.


Adding a new tutorial: spreg
============================

While the :doc:`API docs </library/index>` are automatically generated when
compiling with Sphinx, tutorials that demonstrate use cases for new modules
need to be crafted by the developer. Below we use the case of one particular
module that currently does not have a tutorial as a guide for how to add
tutorials for new modules.

As of PySAL 1.3 there are API docs for
:doc:`spreg </library/spreg/index>`
but no :doc:`tutorial </users/tutorials/index>` currently exists for this module. 

We will fix this and add a tutorial for
:doc:`spreg </library/spreg/index>`.


Requirements
------------

 - sphinx
 - napoleon
 - pysal sources


You can install `sphinx` or `napoleon` using `easy_install` as described
above in :ref:`compiling-doc-label`.

Where to add the tutorial content
---------------------------------

Within the PySAL source the docs live in::

    pysal/doc/source

This directory has the source `reStructuredText`_ files used to render the html
pages. The tutorial pages live under::

    pysal/doc/source/users/tutorials

As of PySAL 1.3, the content of this directory is::

	autocorrelation.rst  fileio.rst  next.rst     smoothing.rst
	dynamics.rst	     index.rst	 region.rst   weights.rst
	examples.rst	     intro.rst	 shapely.rst

The body of the `index.rst` file lists the sections for the tutorials::
	   
	   Introduction to the Tutorials <intro>
	   File Input and Output <fileio>
	   Spatial Weights <weights>
	   Spatial Autocorrelation <autocorrelation>
	   Spatial Smoothing <smoothing>
	   Regionalization <region>
	   Spatial Dynamics <dynamics>
	   Shapely Extension <shapely>
	   Next Steps <next>
	   Sample Datasets <examples>

In order to add a tutorial for `spreg` we need the to change this to read::

	   Introduction to the Tutorials <intro>
	   File Input and Output <fileio>
	   Spatial Weights <weights>
	   Spatial Autocorrelation <autocorrelation>
	   Spatial Smoothing <smoothing>
	   Spatial Regression <spreg>
	   Regionalization <region>
	   Spatial Dynamics <dynamics>
	   Shapely Extension <shapely>
	   Next Steps <next>
	   Sample Datasets <examples>

So we are adding a new section that will show up as `Spatial Regression` and
its contents will be found in the file `spreg.rst`. To create the latter
file simpy copy say `dynamics.rst` to `spreg.rst` and then modify `spreg.rst`
to have the correct content.

Once this is done, move back up to the top level doc directory::

	pysal/doc

Then::

        $ make clean
        $ make html

Point your browser to `pysal/doc/build/html/index.html`

and check your work. You can then make changes to the `spreg.rst` file and
recompile until you are set with the content.

Proper Reference Formatting
---------------------------

For proper hypertext linking of reference material, each unique reference in a
single python module can only be explicitly named once. Take the following example for
instance::

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

It is "named" as "1".  Any other references (even the same paper) with the same "name" will cause a
Duplicate Reference error when Sphinx compiles the document.  Several
work-arounds are available but no concensus has emerged. 

One possible solution is to use an anonymous reference on any subsequent
duplicates, signified by a single underscore with no brackets.  Another solution
is to put all document references together at the bottom of the document, rather
than listing them at the bottom of each class, as has been done in some modules. 



.. _tutorial: /users/tutorials/index
.. _docutils: http://docutils.sourceforge.net/docs/user/tools.html
.. _API docs: /library/index
.. _spreg: /library/spreg/index
.. _Sphinx: http://pypi.python.org/pypi/Sphinx/
.. _here: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _Github: http://github.com/pysal
.. _spreg: /library/spreg/index
.. _reStructuredText: http://sphinx.pocoo.org/rest.html
.. _Sphinx markup guide: http://sphinx.pocoo.org/contents.html
.. _napoleon: http://sphinxcontrib-napoleon.readthedocs.org/en/latest/sphinxcontrib.napoleon.html
.. _readthedocs: http://pysal.readthedocs.org/en/latest
