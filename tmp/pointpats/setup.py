from setuptools import setup
import os.path

from distutils.command.build_py import build_py

with open('README.rst') as file:
    long_description = file.read()

pth = os.path.dirname(os.path.abspath(__file__))+ '/requirements.txt'

REQUIREMENTS = [i.strip() for i in open(pth).readlines()]

Major = 1
Feature = 1
Bug = 0
version = '%d.%d.%d' % (Major, Feature, Bug)

setup(name='pointpats',
      version=version,
      description='Methods and Functions for planar point pattern analysis',
      long_description = long_description,
      url='https://github.com/pysal/pointpats',
      maintainer='Hu Shao',
      maintainer_email='shaohutiger@gmail.com',
      py_modules=['pointpats'],
      python_requires='>3.4',
      test_suite = 'nose.collector',
      tests_require=['nose'],
      keywords='spatial statistics',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
        ],
      license='3-Clause BSD',
      packages=['pointpats'],
      install_requires=REQUIREMENTS,
      zip_safe=False,
      cmdclass = {'build.py':build_py})
