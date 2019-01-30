.. Installation

Installation
============

As of version 2.0.0, PySAL supports python `3.5`_ and `3.6`_ only. Please make sure that you are
operating in a python 3 environment.

Installing the released version
-------------------------------

PySAL is available on the `Python Package Index`_. Therefore, you can either
install directly with `pip` from the command line::

  pip install -U pysal


or download the source distribution (.tar.gz) and decompress it to your selected
destination. Open a command shell and navigate to the decompressed folder.
Type::

  pip install .

Installing development version
------------------------------

Potentially, you might want to use the newest features in the development
version of PySAL on github - `pysal/pysal`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/pysal`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/pysal.git

You can  also `fork`_ the `pysal/pysal`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/PySAL`_, you can
contribute to the PySAL development.

Installing versions supporting Python 2
---------------------------------------

Users requiring Python 2 support can install the legacy version of PySAL: 1.1.14 via pip::

 pip install pysal==1.14.4.post2

Note that this version is only receiving bug fixes. All new enhancements to PySAL are Python 3+ only, and are not available in 1.14.4.

.. _3.5: https://docs.python.org/3.5/
.. _3.6: https://docs.python.org/3.6/
.. _Python Package Index: https://pypi.org/project/PySAL/
.. _pysal/PySAL: https://github.com/pysal/PySAL
.. _fork: https://help.github.com/articles/fork-a-repo/


