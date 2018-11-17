"""Spatial Network Analysis (SPAtial GrapHs: nETworks, Topology, & Inference) This package is part of a refactoring of PySAL. Spaghetti is an open-source python library for the analysis of network-based spatial data. Originating from the network module in PySAL (Python Spatial Analysis Library), it is under active development for the inclusion of newly proposed methods for building graph-theoretic networks and the analysis of network events.
"""

DOCLINES = __doc__.split("\n")

from setuptools import setup
from distutils.command.build_py import build_py


def _get_requirements_from_files(groups_files):
    """returns a dictionary of all requirements keyed by type of requirement.
    Parameters
    ----------
    groups_files : dict
        k - descriptive name, v - file name (including extension)
    Returns
    -------
    groups_reqlist : dict
        k - descriptive name, v - list of required packages
    """
    groups_reqlist = {}
    for k,v in groups_files.items():
        with open(v, 'r') as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list
    return groups_reqlist


def setup_package(package, version):
    """sets up the python package
    Parameters
    ----------
    package : str
        package name
    version : str
        package version info read in from package/__init__.py
    """
    
    _groups_files = {
        'base': 'requirements.txt', #basic requirements
        'tests': 'requirements_tests.txt', #requirements for tests
        'docs': 'requirements_docs.txt', #requirements for building docs
        'plus': 'requirements_plus.txt' #requirements for plus builds
    }
    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop('base')
    extras_reqs = reqs

    setup(name=package,
          version=__version__,
          description=DOCLINES[0],
          url='https://github.com/pysal/'+package,
          download_url='https://pypi.org/project/'+package,
          maintainer='James D. Gaboardi',
          maintainer_email='jgaboardi@gmail.com',
          test_suite = 'nose.collector',
          tests_require=['nose'],
          keywords='spatial statistics, networks, graphs',
          classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: GIS',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6'
            ],
          license='3-Clause BSD',
          packages=[package],
          py_modules=[package],
          install_requires=install_reqs,
          extras_require=extras_reqs,
          zip_safe=False,
          cmdclass = {'build.py':build_py},
          python_requires='>3.4')

if __name__ == '__main__':
    package = 'spaghetti'

    # Get __version__ from package/__init__.py
    with open(package+'/__init__.py', 'r') as f:
        exec(f.readline())

    setup_package(package, __version__)