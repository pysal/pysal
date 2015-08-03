.. _guidelines:

==========
Guidelines
==========
.. contents::

PySAL is adopting many of the conventions in the larger scientific computing
in Python community and we ask that anyone interested in joining the project
please review the following documents:

 * `Documentation standards <http://projects.scipy.org/numpy/wiki/CodingStyleGuidelines>`_
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


You can setup PySAL for local development following the :doc:`installation instructions </users/installation>`.


------------------------
Development Mailing List
------------------------

Development discussions take place on `pysal-dev
<http://groups.google.com/group/pysal-dev>`_.


-----------------------
Release Schedule
-----------------------

PySAL development follows a six-month release schedule that is aligned with
the academic calendar.


1.11 Cycle
==========

========   ========   ================= ====================================================
Start      End        Phase             Notes
========   ========   ================= ====================================================
8/1/15      8/14/15   Module Proposals  Developers draft PEPs and prototype
8/15/15     8/15/15   Developer vote    All developers vote on PEPs 
8/16/15     8/16/15   Module Approval   BDFL announces final approval
8/17/15    12/30/15   Development       Implementation and testing of approved modules
1/1/16       1/1/16   Code Freeze       APIs fixed, bug and testing changes only
1/23/16     1/30/16   Release Prep      Test release builds, updating svn 
1/31/16     1/31/16   Release           Official release of 1.11
========   ========   ================= ====================================================

1.12 Cycle
==========

========   ========   ================= ====================================================
Start      End        Phase             Notes
========   ========   ================= ====================================================
2/1/16      2/14/16   Module Proposals  Developers draft PEPs and prototype
2/15/16     2/15/16   Developer vote    All developers vote on PEPs 
2/16/16     2/16/16   Module Approval   BDFL announces final approval
2/17/16     6/30/16   Development       Implementation and testing of approved modules
7/1/16      7/27/16   Code Freeze       APIs fixed, bug and testing changes only
7/23/16     7/30/16   Release Prep      Test release builds, updating svn 
7/31/16     7/31/16   Release           Official release of 1.12
========   ========   ================= ====================================================



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



