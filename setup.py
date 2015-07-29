# coding: utf-8

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import sys
import shutil
import os
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

from pysal.version import version as dversion

with open('README.txt') as file:
    long_description = file.read()

MAJOR = 1
MINOR = 11
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def setup_package():

    # Perform 2to3 if needed
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))  # get cwd
    src_path = local_path

    if sys.version_info[0] == 3:
        src_path = os.path.join(local_path, 'build', 'py3k')
        sys.path.insert(0, os.path.join(local_path, 'tools'))
        import py3tool
        print("Converting to Python3 via 2to3...")
        py3tool.sync_2to3('pysal', os.path.join(src_path, 'pysal'))

        site_cfg = os.path.join(local_path, 'site.cfg')
        if os.path.isfile(site_cfg):
            shutil.copy(site_cfg, src_path)

        # Ugly hack to make pip work with Python 3, see #1857.
        # Explanation: pip messes with __file__ which interacts badly with the
        # change in directory due to the 2to3 conversion.  Therefore we restore
        # __file__ to what it would have been otherwise.
        global __file__
        __file__ = os.path.join(os.curdir, os.path.basename(__file__))
        if '--egg-base' in sys.argv:
            # Change pip-egg-info entry to absolute path, so pip can find it
            # after changing directory.
            idx = sys.argv.index('--egg-base')
            if sys.argv[idx + 1] == 'pip-egg-info':
                sys.argv[idx + 1] = os.path.join(local_path, 'pip-egg-info')

    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)


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
        version=dversion,
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
            'Programming Language :: Python :: 2.7'
        ],
        packages=find_packages(exclude=[".meta", "*.meta.*", "meta.*",
                                        "meta"]),
        package_data={'pysal': list(example_data_files)},
        requires=['scipy']
    )


if __name__ == '__main__':
    setup_package()
