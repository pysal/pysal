"""
setup.py for PySAL package.
"""

#import distribute_setup
#distribute_setup.use_setuptools()
from distutils.core import setup, Extension
from pysal.version import version

setup(name = 'pysal',
      description = 'PySAL: Python Spatial Analysis Library',
      #long_description = open('INSTALL.txt'),
      maintainer = "PySAL Developers",
	  maintainer_email = "pysal-dev@googlegroups.com",
      url = 'http://pysal.org/',
      download_url = 'http://code.google.com/p/pysal/downloads/list',
      version = version,
      license = 'BSD License',
    packages = ['pysal', 
                'pysal.cg',
                'pysal.cg.tests',
                'pysal.core', 
                'pysal.core.tests', 
                'pysal.core.util', 
                'pysal.core.util.tests', 
                'pysal.core.IOHandlers', 
                'pysal.esda', 
                'pysal.esda.tests', 
                'pysal.inequality',
                'pysal.inequality.tests',
                'pysal.spatial_dynamics',
                'pysal.spatial_dynamics.tests',
                'pysal.spreg',
                'pysal.spreg.tests',
                'pysal.region',
                'pysal.region.tests',
                'pysal.tests',
                'pysal.weights',
                'pysal.weights.tests'],
      

      package_data = {'pysal':['examples/shp_test/*','examples/*.*','examples/README']},

      requires = ['scipy'],

      classifiers=['Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: GIS',])
