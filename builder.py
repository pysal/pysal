'''pysal cross-platform distribution builder script'''
import os, sys

# make docs PDF
os.chdir('doc')
os.system('make clean')
os.system('make latex')
os.chdir('build/latex')
os.system('make all-pdf')
os.chdir('../../../')

#build source and binaries
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

