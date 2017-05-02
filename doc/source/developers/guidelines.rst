.. _guidelines:

==========
Guidelines
==========
.. contents::

PySAL is adopting many of the conventions in the larger scientific computing
in Python community and we ask that anyone interested in joining the project
please review the following documents:

 * `Documentation standards <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
 * `Coding guidelines <http://www.python.org/dev/peps/pep-0008/>`_
 * :doc:`Testing guidelines <testing>`


-----------------------
Open Source Development
-----------------------

PySAL is an open source project and we invite any interested user who wants to
contribute to the project to contact one of the
`team members <https://github.com/pysal?tab=members>`_. For users who
are new to open source development you may want to consult the following
documents for background information:

 * `Contributing to Open Source Projects HOWTO
   <http://www.kegel.com/academy/opensource.html>`_




-----------------------
Source Code
-----------------------


PySAL uses `git <http://git-scm.com/>`_ and github for our  `code repository <https://github.com/pysal/pysal.git/>`_.


Please see `our procedures and policies for development on GitHub <https://github.com/pysal/pysal/wiki/GitHub-Standard-Operating-Procedures>`_
as well as how to `configure your local git for development
<https://github.com/pysal/pysal/wiki/Example-git-config>`_.


You can setup PySAL for local development following the :doc:`installation instructions </users/installation>`.


------------------------
Development Mailing List
------------------------

Development discussions take place on `pysal-dev
<http://groups.google.com/group/pysal-dev>`_
and the `gitter room <https://gitter.im/pysal/pysal>`_.


-----------------------
Release Schedule
-----------------------

As of version 1.11, PySAL has moved to a rolling release model. Discussions
about releases are carried out during the monthly developer meetings and in 
the `gitter room <https://gitter.im/pysal/pysal>`_.


----------
Governance
----------

PySAL is organized around the Benevolent Dictator for Life (BDFL) model of project management.
The BDFL is responsible for overall project management and direction. Developers have a critical role in shaping that
direction. Specific roles and rights are as follows:

=========   ================        ===================================================
Title       Role                    Rights
=========   ================        ===================================================
BDFL        Project Director        Commit, Voting, Veto, Developer Approval/Management
Developer   Development             Commit, Voting
=========   ================        ===================================================

-----------------------
Voting and PEPs
-----------------------

During the initial phase of a release cycle, new functionality for PySAL should be described in a PySAL Enhancment
Proposal (PEP). These should follow the
`standard format  <http://www.python.org/dev/peps/pep-0009/>`_
used by the Python project. For PySAL, the PEP process is as follows

#. Developer prepares a plain text PEP following the guidelines

#. Developer sends PEP to the BDFL

#. Developer posts PEP to the PEP index

#. All developers consider the PEP and vote

#. PEPs receiving a majority approval become priorities for the release cycle



