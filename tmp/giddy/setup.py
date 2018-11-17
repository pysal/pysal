"""GIDDY: GeospatIal Distribution DYnamics

Giddy is an open-source python library for the analysis of dynamics of
longitudinal spatial data. Originating from the spatial dynamics module
in PySAL (Python Spatial Analysis Library), it is under active development
for the inclusion of many newly proposed analytics that consider the
role of space in the evolution of distributions over time and has
several new features including inter- and intra-regional decomposition
of mobility association and local measures of exchange mobility in
addition to space-time LISA and spatial markov methods. Give
giddy a try if you are interested in space-time analysis!

"""

DOCLINES = __doc__.split("\n")

with open('README.rst', 'r', encoding='utf8') as file:
    long_description = file.read()


from setuptools import setup, find_packages
from distutils.command.build_py import build_py
import os

# Get __version__ from giddy/__init__.py without importing the package
# __version__ has to be defined in the first line
with open('giddy/__init__.py', 'r') as f:
    exec(f.readline())

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k,v in groups_files.items():
        with open(v, 'r') as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist

def setup_package():
    _groups_files = {
        'base': 'requirements.txt',
        'tests': 'requirements_tests.txt',
        'docs': 'requirements_docs.txt'
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop('base')
    extras_reqs = reqs

    setup(name='giddy',  # name of package
          version=__version__,
          description=DOCLINES[0],
          #long_description="\n".join(DOCLINES[2:]),
          long_description = long_description,
          url='https://github.com/pysal/giddy',
          maintainer='Wei Kang',
          maintainer_email='weikang9009@gmail.com',
          py_modules=['giddy'],
          python_requires='>3.4',
          test_suite='nose.collector',
          tests_require=['nose'],
          keywords='spatial statistics, spatiotemporal analysis',
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
          packages=find_packages(),
          # packages=['giddy','giddy.tests'],
          install_requires=install_reqs,
          extras_require=extras_reqs,
          zip_safe=False,
          cmdclass={'build.py': build_py})

if __name__ == '__main__':
    setup_package()
