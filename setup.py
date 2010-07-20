"""
setup.py for PySAL package.
"""

#import distribute_setup
#distribute_setup.use_setuptools()
from distutils.core import setup, Extension
from pysal.version import version

setup(name = 'pysal',
      description = 'PySAL: Python Spatial Analysis Library',
      author = 'Luc Anselin, Serge Rey, Charles Schmidt, Andrew Winslow',
      author_email = 'srey@asu.edu',
      url = 'http://pysal.org/',
      version = version,
      packages = ['pysal', 
                  'pysal.cg',
                  'pysal.core', 
                  'pysal.core.IOHandlers', 
                  'pysal.esda', 
                  'pysal.inequality',
                  'pysal.spatial_dynamics',
                  'pysal.region',
                  'pysal.tests',
                  'pysal.weights'],
      package_data = {'pysal':['examples/shp_test/*','examples/*.*','examples/README']},
      requires = ['scipy']
     )
