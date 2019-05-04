# coding: utf-8

from setuptools import setup, find_packages

from distutils.command.build_py import build_py

import os

with open('README.rst') as file:
    long_description = file.read()

with open('pysal/__init__.py', 'r') as f:
    exec(f.readline())


print(find_packages())
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
    example_data_files = set()
    for i in os.listdir("pysal/lib/examples"):
        if i.endswith(('py', 'pyc')):
            continue
        if not os.path.isdir("pysal/lib/examples/" + i):
            if "." in i:
                glob_name = "lib/examples/*." + i.split(".")[-1]
            else:
                glob_name = "lib/examples/" + i
        else:
            glob_name = "lib/examples/" + i + "/*"

        example_data_files.add(glob_name)
    _groups_files = {
        'base': 'requirements.txt',
        'plus': 'requirements_plus.txt',
        'dev': 'requirements_dev.txt',
        'docs': 'requirements_docs.txt'
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop('base')
    extras_reqs = reqs
    setup(
        name='pysal',
        version=__version__,
        description="A library of spatial analysis functions.",
        long_description=long_description,
        maintainer="PySAL Developers",
        maintainer_email='pysal-dev@googlegroups.com',
        url='http://pysal.org',
        download_url='https://pypi.python.org/pypi/pysal',
        license='BSD',
        packages=find_packages(),
        python_requires='>3.5',
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
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ],
        package_data={'pysal': list(example_data_files)},
        install_requires=install_reqs,
        extras_require=extras_reqs,
        cmdclass={'build_py': build_py}
    )


if __name__ == '__main__':
    setup_package()
