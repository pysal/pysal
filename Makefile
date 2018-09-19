# developer Makefile for repeated tasks
# 
.PHONY: clean

download:
	python gitdownload.py

convert:
	python convert.py

test:
	nosetests 

doctest:
	cd doc; make pickle; make doctest

install:
	python setup.py install >/dev/null

src:
	python setup.py sdist >/dev/null

win:
	python setup.py bdist_wininst >/dev/null

prep:
	rm -rf pysal/lib
	mkdir pysal/lib

clean: 
	find . -name "*.pyc" -exec rm '{}' ';'
	find pysal -name "__pycache__" -exec rm -rf '{}' ';'
	rm -rf dist
	rm -rf build
	rm -rf PySAL.egg-info
	rm -rf tmp

