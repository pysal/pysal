'''pysal cross-platform distribution builder script'''
import os, sys

# make docs PDF manually.
# os.chdir('doc')
# os.system('make clean')
# os.system('make latex')
# os.chdir('build/latex')
# os.system('make all-pdf')
# os.chdir('../../../')

#build source and binaries
if sys.platform == 'darwin':
    #source
    os.system('python setup.py sdist --formats=gztar')
    """
    To make a Mac OS X graphical installer, first install 'bdist_mpkg' from
    the Python Package Index using easy_install bdist_mpkg or pip install
    bdist_mpkg. 

    os.system('bdist_mpkg setup.py build')
    
    Then, run bdist_mpkg setup.py build, which builds your package
    installer in the dist/ directory. The mpkg will need to be modified to
    install to the correct location. Edit pysal*.mpkg/Contents/Info.plist
    replacing the path under IFRequirementDicts which a generic System path such
    as,
    /Library/Frameworks/Python.framework/Versions/Current/lib/python2.6/site-packages.
    This will check that directory exists before installing. Note that
    Version/Current/lib/python2.6 is a work around to support both system python
    and EPD. Next find and edit
    pysal*.mpkg/Contents/Packages/pysal*.pkg/Contents/Info.plist. Change the
    value after IFPkgFlagDefaultLocation to point to
    /Library/Frameworks/Python.framework/Versions/Current/lib/python2.6/site-packages

    Next, convert that file (ending in ".mpkg") into a disk image suitable for
    distribution by executing the following command:

    hdiutil create -fs HFS+ -srcfolder pysal*.mpkg/ pysal-x.x.x-py2.X-macosx10.X.dmg

    """

    #binary
    os.system('bdist_mpkg setup.py build')
    # edit the two plist files, then create the dmg next
    #os.system('hdiutil create -fs HFS+ -srcfolder pysal4mac/ pysal4mac.dmg')



elif sys.platform == 'win32':
    #source
    os.system('python setup.py sdist --formats=zip')
    #binary
    os.system('python setup.py bdist --formats=wininst,msi')



elif sys.platform == 'linux2':
    #source
    os.system('python setup.py sdist --formats=gztar')
    #binary
    os.system('python setup.py bdist --formats=rpm')

