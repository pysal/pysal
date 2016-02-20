if [[ $TRAVIS_PYTHON_VERSION == 3* ]]; then
    cd pysal; python -m 'nose';
else
    nosetests --with-coverage --cover-package=pysal; 
fi
