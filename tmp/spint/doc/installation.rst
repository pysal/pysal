.. Installation

Installation
============


spint supports python `3.5`_ and `3.6`_ only. Please make sure that you are
operating in a python 3 environment.

Installing released version
---------------------------

spint is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U spint


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of spint on github - `pysal/spint`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/spint`_
by running the following from a command shell::

  pip install https://github.com/pysal/spint/archive/master.zip

You can  also `fork`_ the `pysal/spint`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/spint`_, you can
contribute to the mgwr development.

.. _3.5: https://docs.python.org/3.5/
.. _3.6: https://docs.python.org/3.6/
.. _Python Package Index: https://pypi.org/project/spglm/
.. _pysal/spint: https://github.com/pysal/spint
.. _fork: https://help.github.com/articles/fork-a-repo/
