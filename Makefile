# developer Makefile for repeated tasks
#

# read arguments from modules.txt
$SUBMODULES = asdf

.PHONY: clean

#test:
#	nosetests 
#
#doctest:
#	cd doc; make pickle; make doctest
#
#install:
#	python setup.py install >/dev/null
#
#src:
#	python setup.py sdist >/dev/null
#
#win:
#	python setup.py bdist_wininst >/dev/null
#
clean: 
	find . -name "*.pyc" -exec rm '{}' ';'
	find pysal -name "__pycache__" -exec rm -rf '{}' ';'
	rm -rf dist
	rm -rf build
	rm -rf PySAL.egg-info

reorg_dist:
	find ./pysal/ -type d -delete	
	for (i in 1:modules):
		git clone $module pysal/
		#grab meta info about commits vs. last release
		rm -r pysal/$module/.git
		$module/MANIFEST.in #exclude submodule-specific files #optional
		git add $module/.
	git add .

reorg_dev:
	find ./pysal/ -type d -delete	
	for (i in 1:modules):
		git clone $module pysal/
		cd module $module && git remote rename origin upstream
		git remote add origin git@github.com:$git_user/$module
	mv .git .git_bak

undo_dev:
	mv .git .git_bak2
	if [[ isfile ]]{
		mv .git_bak .git
	}
