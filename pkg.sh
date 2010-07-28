#!/bin/bash

#uncomment next to build fresh PDF of docs
#cd doc/;make latex;cd build/latex;make all-pdf;cd ../../../

#here we build the distros

platform=$(uname)
if [[ $platform == 'Darwin' ]]; then
    python setup.py sdist --formats=gztar,zip
    python setup.py bdist --formats=wininst
    bdist_mpkg setup.py build
elif [[ $platform == 'win32' ]]; then
    python setup.py bdist --formats=wininst,msi
elif [[ $platform == 'Linux' ]]; then
    python setup.py bdist --formats=rpm
fi
rm -rf build/
