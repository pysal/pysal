#!/bin/bash

cd doc/
make latex
cd build/latex
make all-pdf
cd ../../../
python setup.py sdist --formats=gztar,zip
python setup.py bdist_wininst
rm -rf build/
