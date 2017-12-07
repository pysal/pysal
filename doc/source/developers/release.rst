.. _release:
.. role:: strike

************************
PySAL Release Management
************************
.. contents::

Prepare the release
-------------------

- Check all tests pass. See :doc:`testing`.
- Update CHANGELOG::

     $ python tools/github_stats.py days >> chglog

- where `days` is the number of days to start the logs at.
- Prepend `chglog` to `CHANGELOG` and edit.
- Edit THANKS and README and README.md if needed.
- Edit the file `version.py` to update MAJOR, MINOR, MICRO.
- Bump::

     $ cd tools; python bump.py

- Commit all changes.
- Push_ your branch up to your GitHub repos.
- On github issue a pull request, with a target of **upstream master**. 
  Add a comment that this is for release.

Make docs
---------

As of version 1.6, docs are automatically compiled and hosted_.

Make a source dist and test locally (Python 3)
----------------------------------------------

On each build machine::

     $  git clone http://github.com/pysal/pysal.git
     $  cd pysal
     $  python setup.py sdist
     $  cp dist/PySAL-1.14.2.tar.gz ../junk/.
     $  cd ../junk
     $  conda create -n pysaltest3 python=3 pip nose nose-progressive nose-exclude
     $  source activate pysaltest3
     $  pip install PySAL-1.14.2.tar.gz
     $  rm -r /home/serge/anaconda3/envs/pysaltest3/lib/python3.6/site-packages/pysal/contrib
     $  nosetests /home/serge/anaconda3/envs/pysaltest3/lib/python3.6/site-packages/pysal/

You can modify the above to test for Python 2 environments.


Upload release to pypi
----------------------

- Make and upload_ to the `Python Package Index`_ in one shot!::

     $ python setup.py sdist upload

  - if not registered_, do so by following the prompts::

     $ python setup.py register

  - You can save the login credentials in a dot-file, `.pypirc`.

- Make and upload the Windows installer to SourceForge_. On a Windows box, build the installer as so::

     $ python setup.py bdist_wininst

Create a release on github
--------------------------

https://help.github.com/articles/creating-releases/


Announce
--------

- Draft and distribute press release on openspace-list, pysal.org, `spatial.ucr.edu`_ .


Bump master version
-------------------

- Change MAJOR, MINOR version in `setup.py`.
- Change `pysal/version.py`.
- Change the docs version by editing `doc/source/conf.py` in two places (version and release).
- Update the release schedule in `doc/source/developers/guidelines.rst`.


Update the `github.io news page <https://github.com/pysal/pysal.github.io/blob/master/_includes/news.md>`_
to announce the release.

.. _upload: https://docs.python.org/2.7/distutils/packageindex.html#the-upload-command
.. _registered: https://docs.python.org/2.7/distutils/packageindex.html#the-register-command
.. _source: http://docs.python.org/distutils/sourcedist.html
.. _hosted: http://pysal.readthedocs.io/en/latest/users/index.html
.. _branch: https://github.com/pysal/pysal/wiki/GitHub-Standard-Operating-Procedures
.. _policy: https://github.com/pysal/pysal/wiki/Example-git-config
.. _create the release: https://help.github.com/articles/creating-releases/
.. _Push: https://github.com/pysal/pysal/wiki/GitHub-Standard-Operating-Procedures
.. _Python Package Index: https://pypi.python.org/pypi/PySAL
.. _SourceForge: https://sourceforge.net
.. _spatial.ucr.edu: http://spatial.ucr.edu/news.html
