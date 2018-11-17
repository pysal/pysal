# coding: utf-8

from setuptools import setup, find_packages

from distutils.command.build_py import build_py

import os

with open('README.rst', 'r', encoding='utf8') as file:
    long_description = file.read()

# Get __version__ from libpysal/__init__.py without importing the package
# __version__ has to be defined in the first line
with open('libpysal/__init__.py', 'r') as f:
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
    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    _groups_files = {
        'base': 'requirements.txt',
        'plus': 'requirements_plus.txt',
        'dev': 'requirements_dev.txt'
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop('base')
    extras_reqs = reqs

    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    example_data_files = set()
    for i in os.listdir("libpysal/examples"):
        if i.endswith(('py', 'pyc')):
            continue
        if not os.path.isdir("libpysal/examples/" + i):
            if "." in i:
                glob_name = "examples/*." + i.split(".")[-1]
            else:
                glob_name = "examples/" + i
        else:
            glob_name = "examples/" + i + "/*"

        example_data_files.add(glob_name)

    setup(
        name='libpysal',
        version=__version__,
        description="Core components of PySAL A library of spatial analysis functions.",
        long_description=long_description,
        maintainer="PySAL Developers",
        maintainer_email='pysal-dev@googlegroups.com',
        url='http://pysal.org',
        download_url='https://pypi.python.org/pypi/libpysal',
        license='BSD',
        py_modules=['libpysal'],
        packages=find_packages(),
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
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6'
        ],
        package_data={'libpysal':list(example_data_files)},
        install_requires=install_reqs,
        extras_require=extras_reqs,
        cmdclass={'build_py': build_py},
        python_requires='>3.4'
    )


if __name__ == '__main__':
    setup_package()
