# coding: utf-8

from setuptools import setup, find_packages

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

import os

with open('README.rst') as file:
    long_description = file.read()

MAJOR = 1
MINOR = 12
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def setup_package():

    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    example_data_files = set()
    for i in os.listdir("pysal/examples"):
        if i.endswith(('py', 'pyc')):
            continue
        if not os.path.isdir("pysal/examples/" + i):
            if "." in i:
                glob_name = "examples/*." + i.split(".")[-1]
            else:
                glob_name = "examples/" + i
        else:
            glob_name = "examples/" + i + "/*"

        example_data_files.add(glob_name)

    setup(
        name='PySAL',
        version=VERSION,
        description="A library of spatial analysis functions.",
        long_description=long_description,
        maintainer="PySAL Developers",
        maintainer_email='pysal-dev@googlegroups.com',
        url='http://pysal.org',
        download_url='https://pypi.python.org/pypi/PySAL',
        license='BSD',
        py_modules=['pysal'],
        test_suite='nose.collector',
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
            'Programming Language :: Python :: 2.5',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4'
        ],
        packages=find_packages(exclude=[".meta", "*.meta.*", "meta.*",
                                        "meta"]),
        package_data={'pysal': list(example_data_files)},
        install_requires=[
            'scipy>=0.11',
            'numpy>=1.3',
        ],
        extras_require={
            'plus': [
                'matplotlib>=1.5.1',
                'seaborn>=0.7.0',
                'geopandas>=0.2',
                'scikit-learn>=0.17.1',
                'bokeh>=0.11.1',
                'geojson>=1.3.2',
                'folium>=0.2.1',
                'mplleaflet>=0.0.5',
                'statsmodels>=0.6.1',
                'numba',
                'numexpr',
            ],
            'dev': [
                'nose',
                'nose-progressive',
                'nose-exclude',
                'coverage',
                'sphinx',
                'sphinxcontrib-napoleon',
                'coveralls',
            ],
        },
        cmdclass={'build_py': build_py}
    )


if __name__ == '__main__':
    setup_package()
