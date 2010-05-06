"""
setup.py for PySAL package.
"""

from distutils.core import setup, Extension
from pysal.version import version

setup(name = 'pysal',
      description = 'PySAL: Python Spatial Analysis Library',
      author = 'Luc Anselin, Serge Rey, Charles Schmidt, Andrew Winslow',
      url = 'http://pysal.org/',
      version = version,
      packages = ['pysal', 
                  'pysal.cg',
                  'pysal.core', 
                  'pysal.core.IOHandlers', 
                  'pysal.econometrics',
                  'pysal.esda', 
                  'pysal.inequality',
                  'pysal.spatial_dynamics',
                  'pysal.region',
                  'pysal.weights'],
      package_data = {'pysal':['examples/shp_test/*','examples/*.*','examples/README']},
      requires = ['scipy']
     )
