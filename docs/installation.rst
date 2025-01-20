.. Installation

Installation
============


You can install the meta package `pysal` which results in all the packages in
the `pysal` ecosystem being installed.

Alternatively, individual packages can be installed a la carte.


Installing the pysal meta package
---------------------------------
.. tab-set-code::

    .. code-block:: pip

        pip install pysal

    .. code-block:: conda

	conda install --channel conda-forge pysal

Installing individual pysal packages
------------------------------------
A similar approach can be used to select one, or a subset, of pysal packages
without having to install the entire ecosystem. For example, to install `esda`
and `mapclassify`

.. tab-set-code::

    .. code-block:: pip

        pip install esda mapclassify

    .. code-block:: conda

	conda install --channel conda-forge esda mapclassify

   

Installing the development version
----------------------------------

Potentially, you might want to use the newest features in the development
version of PySAL on github - `pysal/pysal`_ while have not been incorporated
in the Pypi released version. You can achieve that by installing `pysal/pysal`_
by running the following from a command shell::

  pip install git+https://github.com/pysal/pysal.git

You can  also `fork`_ the `pysal/pysal`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/PySAL`_, you can
contribute to the PySAL development.


.. _dependencies:

Dependencies
------------

Required
++++++++
- `geopandas`_
- `seaborn`_
- `descartes`_
- `palettable`_
- `scikit-learn`_
- `rtree`_
- `tqdm`_
- `statsmodels`_
- `deprecated`_


Optional
++++++++
- `urbanaccess`_
- `pandana`_
- `numba`_
- `numexpr`_
- `bokeh`_




Installing versions supporting Python 2
---------------------------------------

Users requiring Python 2 support can install the legacy version of PySAL: 1.1.14 via pip::

 pip install pysal==1.14.4.post2

Note that this version is only receiving bug fixes. All new enhancements (post 2019-01) to PySAL are Python 3+ only, and are not available in 1.14.4.

.. _3.7: https://docs.python.org/3.7/
.. _3.6: https://docs.python.org/3.6/
.. _Python Package Index: https://pypi.org/project/PySAL/
.. _pysal/PySAL: https://github.com/pysal/PySAL
.. _conda: https://docs.conda.io/en/latest/
.. _Anaconda Distribution: https://docs.continuum.io/anaconda/
.. _fork: https://help.github.com/articles/fork-a-repo/
.. _geopandas: http://geopandas.org/install.html
.. _seaborn: https://seaborn.pydata.org/installing.html
.. _descartes: https://pypi.org/project/descartes/
.. _palettable: https://jiffyclub.github.io/palettable/
.. _scikit-learn: https://scikit-learn.org/stable/install.html
.. _rtree: http://toblerity.org/rtree/install.html
.. _tqdm: https://pypi.org/project/tqdm/
.. _statsmodels: https://www.statsmodels.org/stable/install.html
.. _deprecated: https://pypi.org/project/Deprecated/
.. _urbanaccess: https://github.com/UDST/urbanaccess
.. _pandana: https://pypi.org/project/pandana/ 
.. _numba: https://numba.pydata.org/numba-doc/dev/user/installing.html
.. _numexpr: https://pypi.org/project/numexpr/
.. _bokeh: https://bokeh.pydata.org/en/latest/docs/installation.html
