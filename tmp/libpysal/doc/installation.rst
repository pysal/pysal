.. Installation

Installation
============

libpysal supports python `3.5`_ and `3.6`_ only. Please make sure that you are
operating in a python 3 environment.

Installing released version
---------------------------

libpysal is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U libpysal


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of libpysal on github - `pysal/libpysal`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/libpysal`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/libpysal.git

You can  also `fork`_ the `pysal/libpysal`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/libpysal`_, you can
contribute to libpysal development.

.. _3.5: https://docs.python.org/3.5/
.. _3.6: https://docs.python.org/3.6/
.. _Python Package Index: https://pypi.org/project/libpysal/
.. _pysal/libpysal: https://github.com/pysal/libpysal
.. _fork: https://help.github.com/articles/fork-a-repo/
