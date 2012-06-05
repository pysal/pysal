#!/bin/bash - 
#===============================================================================
#
#          FILE:  pysal-run-tests.sh
# 
#         USAGE:  ./pysal-run-tests.sh 
# 
#   DESCRIPTION:  
# 
#       OPTIONS:  ---
#  REQUIREMENTS:  ---
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR: PHILIP STEPHENS (), phil.stphns@gmail.com
#       COMPANY: GeoDa Center for Geospatial Analysis and Computation
#       CREATED:  ---
#      MODIFIED: 09/09/2011 12:52:16 MST
#      REVISION:  ---
#===============================================================================
set -o nounset   # Treat unset variables as an error

mkdir -p /tmp/testspace/
rm -rf /tmp/testspace/*
cd /tmp/testspace/
svn checkout http://pysal.googlecode.com/svn/trunk/ 
export PYTHONPATH=/tmp/testspace/pysal/trunk/:$PYTHONPATH
cd trunk
b=$(svnversion)

# -----------------------------setup message header
echo "to: pastephe@asu.edu" > /tmp/report.txt
echo "from: phil.stphns@gmail.com" >> /tmp/report.txt
printf "Subject: Testing pysal test messaging --> for revision $b" >> /tmp/report.txt
echo "" >> /tmp/report.txt 
#-----------------------------

echo "PySAL Test Suite Report" >> /tmp/report.txt 
echo "=======================" >> /tmp/report.txt 
svn log -r $(svnversion) >> /tmp/report.txt 
/usr/local/bin/python -c 'import os,sys, numpy, scipy; print os.name; print sys.platform, sys.version; print "Scipy version:", scipy.__version__; print "Numpy version:", numpy.__version__' >> /tmp/report.txt
echo "-----------------------" >> /tmp/report.txt 
echo "" >> /tmp/report.txt 
echo "" >> /tmp/report.txt 
echo "Last Unittest Report     https://dl.dropbox.com/u/153148/public_html/tests/tests.txt" >> /tmp/report.txt 
echo "Last Doctest  Report     https://dl.dropbox.com/u/153148/public_html/tests/tutorials.txt" >> /tmp/report.txt 
echo "Last Coverage Report     https://dl.dropbox.com/u/153148/public_html/coverage/index.html" >> /tmp/report.txt 
echo "Nightly Builds           http://pysal.geodacenter.org/builds/" >> /tmp/report.txt
echo "Nightly Windows Report   https://dl.dropbox.com/u/153148/public_html/tests/windowstests.txt" >> /tmp/report.txt 
echo "" >> /tmp/report.txt 
echo "" >> /tmp/report.txt 


#/usr/local/share/python/nosetests --quiet  --with-coverage --cover-html --cover-package=pysal --cover-html-dir=/Users/stephens/Dropbox/Public/public_html/coverage  pysal/ >> /Users/stephens/Dropbox/Public/public_html/tests/tests.txt 2>&1 


echo "" >> /tmp/report.txt 
echo "Nose Test Summary" >> /tmp/report.txt 
tail -4 /Users/stephens/Dropbox/Public/public_html/tests/tests.txt >> /tmp/report.txt


cd doc

#/usr/local/share/python/sphinx-build -b doctest -d build/doctrees  source build/doctest >> /Users/stephens/Dropbox/Public/public_html/tests/tutorials.txt  2>/dev/null

tail /Users/stephens/Dropbox/Public/public_html/tests/tutorials.txt >> /tmp/report.txt

echo "-----------------------" >> /tmp/report.txt 
echo "End of Report" >> /tmp/report.txt 



# remove instances of a single period on a line which causes sendmail to send now
sed "s/^\./\.\./g" /tmp/report.txt > /tmp/report.eml
/usr/sbin/sendmail -t < /tmp/report.eml

