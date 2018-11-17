#!/bin/bash - 

if [ -f /tmp/pysal.lock ]
then
#check repo for changes
cd /home/pstephen/pysal/
svn cleanup
a=$(svnversion)
svn update -r $((a+1)) 2>&1

if [ $? = "0" ]
then
b=$(svnversion) 
else exit
fi

if [ "$a" != "$b"  ]
then 
export PYTHONPATH=/home/pstephen/pysal/
find /home/pstephen/pysal/pysal -name "*.pyc" -exec rm '{}' ';'

# run coverage and copy to server
/usr/local/bin/coverage html -d /home/pstephen/coverage pysal/*.py pysal/cg/*.py pysal/esda/*.py pysal/inequality/*.py pysal/network/*.py pysal/region/*.py pysal/spatial_dynamics/*.py pysal/spreg/*.py pysal/weights/*.py pysal/core/*.py pysal/core/IOHandlers/*.py pysal/core/util/*.py 
rsync -r --delete /home/pstephen/coverage/ stephens@geodacenter.org:~/coverage

# build new docs and copy to server
cd /home/pstephen/pysal/doc
/usr/bin/make clean
/usr/bin/sphinx-build -Q -b html  -d build/doctrees  source build/html 
rsync -r --delete /home/pstephen/pysal/doc/build/html/ stephens@geodacenter.org:~/dev

# build source installer and copy to server
cd /home/pstephen/pysal/
/usr/bin/python setup.py sdist
rsync -r --delete /home/pstephen/pysal/dist/pysal*  stephens@geodacenter.org:~/tmp/builds

else echo "`date`, "$a", "$b", Repo unchanged." > /tmp/pysal.log 2>&1
fi
fi
