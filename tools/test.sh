#!/bin/bash - 

if [ -f /tmp/pysal.lock ]
    then
    cd /Users/stephens/tmp/pysal/
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
        #cd pysal
        #rm -rf /tmp/pysal
        #svn checkout http://pysal.googlecode.com/svn/trunk pysal
        export PYTHONPATH=/Users/stephens/tmp/pysal
        export PATH=/Library/Frameworks/EPD64.framework/Versions/Current/bin:$PATH
        find pysal -name "*.pyc" -exec rm '{}' ';'

        # setup message header
        #echo "to: pas@asu.edu" > /tmp/report.txt
        echo "to: phil.stphns@gmail.com" > /tmp/report.txt
        echo "from: phil.stphns@gmail.com" >> /tmp/report.txt
        printf "Subject: PySAL Unittest Results for revision $b" >> /tmp/report.txt
        echo "" >> /tmp/report.txt 


        cd /Users/stephens/tmp/pysal/
        svn log -r $(svnversion) >> /tmp/report.txt 
        echo "" >> /tmp/report.txt 

        # print system information
        python -c 'import os,sys, numpy, scipy; print sys.platform, sys.version; print "Scipy version:", scipy.__version__; print "Numpy version:", numpy.__version__' >> /tmp/report.txt
        echo "" >> /tmp/report.txt 

        echo "Full Coverage Report --> http://pysal.geodacenter.org/coverage/index.html" >> /tmp/report.txt
        echo "" >> /tmp/report.txt 

        # execute pep8 stats
        echo "" >> /tmp/report.txt 
        echo "PEP 8 Stats" >> /tmp/report.txt 
        pep8 --statistics -qq . >> /tmp/report.txt
        echo "" >> /tmp/report.txt 

        # execute nose test framework
        nosetests pysal/ >> /tmp/report.txt 2>&1
        #nosetests --with-coverage --cover-html --cover-package=pysal --cover-html-dir=/tmp/coverage pysal/ >> /tmp/report.txt 2>&1
        #rsync -r --delete /tmp/coverage/ stephens@geodacenter.org:~/coverage
        echo "" >> /tmp/report.txt 

        # execute sphinx doctest framework
        cd /Users/stephens/tmp/pysal/doc/
        /usr/bin/make clean
        sphinx-build -b doctest -d build/doctrees  source build/doctest >> /tmp/report.txt  2>/dev/null
        echo "" >> /tmp/report.txt 


        # remove instances of a single period on a line which causes sendmail to send now
        sed "s/^\./\.\./g" /tmp/report.txt > /tmp/report.eml
        /usr/sbin/sendmail -t < /tmp/report.eml

    else echo "`date`, "$a", "$b", Repo unchanged." > /tmp/pysal.log 2>&1
  fi
fi
