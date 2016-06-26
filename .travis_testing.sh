if [[ $TRAVIS_PYTHON_VERSION == 3* ]]; then
    nosetests --with-coverage --cover-package=pysal --exclude-dir=pysal/contrib
else
    nosetests --with-coverage --cover-package=pysal; 
fi
