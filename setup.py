"""
setup.py for PySAL package.
"""

DOCLINES = __doc__.split("\n")

import sys
import subprocess
import re
import shutil
import os
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
#import distribute_setup
#distribute_setup.use_setuptools()
#from distutils.core import setup 
from setuptools import setup, find_packages
from pysal.version import version as dversion
#version = '1.3.x'

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: BSD License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: GIS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

NAME                = 'pysal'
MAINTAINER          = "PySAL Developers"
MAINTAINER_EMAIL    = "pysal-dev@googlegroups.com"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://pysal.org"
#DOWNLOAD_URL        = 'http://code.google.com/p/pysal/downloads/list'
DOWNLOAD_URL        = 'http://pysal.googlecode.com/files/pysal-1.5.0.zip'
LICENSE             = 'BSD'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR               = 1
MINOR               = 4
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
def setup_package():

    # Perform 2to3 if needed
    local_path = os.path.dirname(os.path.abspath(sys.argv[0])) #get cwd
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

    # Rewrite the version file everytime
    #write_version_py()
    
    setup(
        name = NAME,
        description = DESCRIPTION,
        long_description = "\n".join(DOCLINES[2:]),
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        url = URL,
        download_url = DOWNLOAD_URL,
        version = dversion,
        license = LICENSE,
        test_suite = 'nose.collector',
        classifiers = CLASSIFIERS,
        packages = ['pysal', 
            'pysal.cg',
            'pysal.cg.tests',
            'pysal.contrib', 
            'pysal.contrib.weights_viewer', 
            'pysal.core', 
            'pysal.core.tests', 
            'pysal.core.util', 
            'pysal.core.util.tests', 
            'pysal.core.IOHandlers', 
            'pysal.esda', 
            'pysal.esda.tests', 
            'pysal.examples', 
            'pysal.inequality',
            'pysal.inequality.tests',
            'pysal.spatial_dynamics',
            'pysal.spatial_dynamics.tests',
            'pysal.spreg',
            'pysal.spreg.tests',
            'pysal.region',
            'pysal.region.tests',
            'pysal.weights',
            'pysal.weights.tests'],
        package_data = {'pysal':['examples/*'] },
        requires = ['scipy'],
        )
            

if __name__ == '__main__':
    setup_package()
