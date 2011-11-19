from setuptools import setup, find_packages
from pysal.version import version

setup(
   name = 'pySAL',
   version = version,
   packages = find_packages(),
   scripts = [],
   install_requires = ['scipy>=0.7'],
   package_data = {'doc': ['build/latex/pysal.pdf'], 'pysal':['examples/shp_test/*','examples/*.*','examples/README']},
   test_suite = 'pysal.tests.tests' ,
   use_2to3 = True,

   maintainer = "PySAL Developers",
   maintainer_email = "pysal-dev@googlegroups.com",
   description = 'PySAL: Python Spatial Analysis Library',
   license = 'BSD License',
   url = 'http://pysal.org/',
   download_url = 'http://code.google.com/p/pysal/downloads/list',
   classifiers = ['Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: GIS']
          
   )
