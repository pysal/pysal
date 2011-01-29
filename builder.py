# pysal cross-platform distribution builder script
'''
#uncomment next to build fresh PDF of docs
#cd doc/;make latex;cd build/latex;make all-pdf;cd ../../../
'''
import os, sys
if sys.platform == 'darwin':
    os.system('python setup.py sdist --formats=gztar,zip')
    os.system('python setup.py bdist --formats=wininst')
    os.system('bdist_mpkg setup.py build')
    os.system('mkdir dist/pysal4mac')
    os.system('mv dist/*.mpkg dist/pysal4mac')
    os.chdir('dist')
    os.system('pwd')
    os.system('hdiutil create -fs HFS+ -srcfolder pysal4mac/ pysal4mac.dmg')
    os.system('rm -rf pysal4mac/')
    os.system('rm -rf ../build/')

elif sys.platform == 'win32':
    os.system('python setup.py sdist --formats=gztar,zip')
    os.system('python setup.py bdist --formats=wininst,msi')
    os.system('rm -rf build/')

elif sys.platform == 'linux2':
    os.system('python setup.py sdist --formats=gztar,zip')
    os.system('python setup.py bdist --formats=rpm')
    os.system('rm -rf build/')

