.. _testing:

************************
PySAL Testing Procedures
************************

All public classes and functions should include examples in their docstrings. Those examples serve two purposes:

#. Documentation for users
#. Tests to ensure code behavior is aligned with the documentation

To ensure that any changes made to one module do not introduce breakage in the wider project, developers should run the
package wide test script in `tests/tests.py <http://code.google.com/p/pysal/source/browse/trunk/pysal/tests/tests.py>`_ before making any commits.

The trunk should most always be in a state where all tests are passed.


Package-wide tests for PySAL
============================


Currently allows for testing of docstring examples from individual modules.
Prior to commiting any changes to the trunk, said changes should be checked
against the rest of the local copy of the trunk by running::

    python tests.py

If all tests pass, changes have not broken the current trunk and can be
committed. Commits that introduce breakage should only be done in cases where
other developers are notified and the breakage raises important issues for
discussion.


Notes
=====
New modules need to be included in the `#module imports` section, as
well as in the truncated module list where `mods` is first defined.

To deal with relative paths in the doctests a symlink must first be made from
within the `tests` directory as follows::

     ln -s ../examples .


