.. _release:
.. role:: strike

************************
PySAL Release Management
************************
.. contents::

Prepare the release
-------------------

- Create a branch_.
- Check all tests pass. See :doc:`testing`.
- Update CHANGELOG::

     $ python tools/github_stats.py >> chglog

- Prepend `chglog` to `CHANGELOG` and edit
- Edit THANKS and README and README.md if needed.
- Change MAJOR, MINOR version in setup script.
- Change pysal/version.py to non-dev number
- Change the docs version from X.xdev to X.x by editing doc/source/conf.py in two places.
- Change docs/index.rst to update Stable version and date, and Development version
- Commit all changes.
- Push_ your branch up to your GitHub repos
- On github issue a pull request. Add a comment that this is for release.


Tag 
---

With version 1.10 we changed the way the tags are created to reflect our
policy_ of not pushing directly to upstream.

After you have issued a pull request, the project maintainer can `create the release`_ on GitHub. 


Make docs
---------

As of version 1.6, docs are automatically compiled and hosted_.

Make and Upload distributions
-------------------------------

On each build machine, clone and checkout the newly created tag (assuming that
is `v1.10` in what follows)::

  $ git clone http://github.com/pysal/pysal.git
  $ cd pysal
  $ git fetch --tags
  $ git checkout v1.10

- Make and upload_ to the Python Package Index in one shot!::

   $ python setup.py sdist  (to test it)
   $ python setup.py sdist upload

  - if not registered_, do so. Follow the prompts. You can save the
      login credentials in a dot-file, .pypirc

- Make and upload the Windows installer to SourceForge.
  - On a Windows box, build the installer as so:: 

    $ python setup.py bdist_wininst

Announce
--------

- Draft and distribute press release on geodacenter.asu.edu, openspace-list, and pysal.org

  - On GeoDa center website, do this:

   - Login and expand the wrench icon to reveal the Admin menu
   - Click "Administer", "Content Management", "Content"
   - Next, click "List", filter by type, and select "Featured Project".
   - Click "Filter"

   Now you will see the list of Featured Projects. Find "PySAL".

   - Choose to 'edit' PySAL and modify the short text there. This changes the text users see on the homepage slider.
   - Clicking on the name "PySAL" allows you to edit the content of the PySAL project page, which is also the "About PySAL" page linked to from the homepage slider.

Put master back to dev
----------------------

- Change MAJOR, MINOR version in setup script.
- Change pysal/version.py to dev number
- Change the docs version from X.x to X.xdev by editing doc/source/conf.py in two places.
- Update the release schedule in doc/source/developers/guidelines.rst


Update the `github.io news page <https://github.com/pysal/pysal.github.io/blob/master/_includes/news.md>`_
to  announce the release.

.. _upload: http://docs.python.org/2.7/distutils/uploading.html
.. _registered: http://docs.python.org/2.7/distutils/packageindex.html
.. _source: http://docs.python.org/distutils/sourcedist.html
.. _hosted: http://pysal.readthedocs.org
.. _branch: https://github.com/pysal/pysal/wiki/GitHub-Standard-Operating-Procedures
.. _policy: https://github.com/pysal/pysal/wiki/Example-git-config
.. _create the release: https://help.github.com/articles/creating-releases/
.. _Push: https://github.com/pysal/pysal/wiki/GitHub-Standard-Operating-Procedures
