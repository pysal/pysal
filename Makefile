# developer Makefile for repeated tasks
# 
.PHONY: clean

test:
	nosetests pysal

src:
	python setup.py sdist

win:
	python setup.py bdist_wininst

clean: 
	find pysal -name "*.pyc" -exec rm '{}' ';'
	find pysal -name "__pycache__" -exec rm -rf '{}' ';'
