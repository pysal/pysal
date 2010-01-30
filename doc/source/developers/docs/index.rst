****************************
Building PySAL Documentation
****************************

PySAL documentation is built using `Sphinx <http://sphinx.pocoo.org/>`_.
We also use the `numpydoc <http://pypi.python.org/pypi/numpydoc/0.2>`_ extension to Sphinx which customizes the markup
of the docstrings of the PySAL modules.

The source for the docs is in `trunk/doc`. Building the documentation is
done as follows (assuming
sphinx and numpydoc are already installed)::

        serge@think:~/Research/p/PySAL/src/google/trunk/doc$ ls
        build  Makefile  source

        serge@think:~/Research/p/PySAL/src/google/trunk/doc$ make clean;make html


To see the results in a browser open `build/html/index.html`. To make
changes, edit (or add) the relevant files in `source` and rebuild the
docs. Consult the `Sphinx markup guide <http://sphinx.pocoo.org/contents.html>`_ for details on the syntax and structure of the files in `source`.

Once you are happy with your changes, check-in the `source` files, do not
add or check-in files under  `build` since they are dynamically built.


Changes added to the `svn repository <http://code.google.com/p/pysal/>`_
will be propogated to `pysal.org <http://pysal.org>`_ on an hourly basis.




