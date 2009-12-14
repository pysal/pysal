"""
setup.py for PySAL package.
"""

from distutils.core import setup, Extension
from pysal.version import version

setup(name = 'pysal',
      description = 'PySAL: Python Spatial Analysis Library',
      author = 'Luc Anselin, Serge Rey, Charles Schmidt, Andrew Winslow',
      url = 'http://pysal.org/',
      version = version
      packages = ['pysal', 
                  'pysal.cg',
                  'pysal.core', 
                  'pysal.core._FileIO', 
                  'pysal.core._FileIO.pyshp', 
                  'pysal.econometrics',
                  'pysal.esda', 
                  'pysal.inequality',
                  'pysal.markov',
                  'pysal.region',
                  'pysal.weights'],
      ext_modules = [Extension('pysal.core._FileIO.pyshp._shp', ['pysal/core/_FileIO/pyshp/_shp.c'])],
      package_data = {'pysal':['examples/shp_test/*','examples/*.*','examples/README']},
      requires = ['scipy']
     )
