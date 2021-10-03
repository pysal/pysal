# coding: utf-8
import codecs
import os.path
import os
import versioneer
from setuptools import setup, find_packages
from distutils.command.build_py import build_py

with open('README.md') as file:
    long_description = file.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


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
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass({"build_py": build_py}),
        description="A library of spatial analysis functions.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        maintainer="PySAL Developers",
        maintainer_email='pysal-dev@googlegroups.com',
        url='http://pysal.org',
        download_url='https://pypi.python.org/pypi/pysal',
        license='BSD',
        packages=find_packages(),
        python_requires='>3.6',
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
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
        ],
        install_requires=install_reqs,
        extras_require=extras_reqs,
    )


if __name__ == '__main__':
    setup_package()
