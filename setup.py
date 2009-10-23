"""
setup.py for PySAL package.
"""

from distutils.core import setup, Extension

setup(name = 'pysal',
      description = 'PySAL: Python Spatial Analysis Library',
      author = 'Luc Anselin, Serge Rey, Charles Schmidt, Andrew Winslow',
      url = 'http://pysal.org/',
      version = '0.1',
      packages = ['pysal', 
                  'pysal.cg',
                  'pysal.core', 
                  'pysal.core._FileIO', 
                  'pysal.core._FileIO.pyshp', 
                  'pysal.esda', 
                  'pysal.inequality',
                  'pysal.weights'],
      ext_modules = [Extension('pysal.core._FileIO.pyshp._shp', ['pysal/core/_FileIO/pyshp/_shp.c'])],
      package_data = {'pysal':['examples/shp_test/*','examples/*.*','examples/README']},
      requires = ['scipy']
     )

