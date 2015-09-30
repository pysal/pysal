import copy
import pysal.spreg as sr
import patsy as p
from numpy import array, ndarray, asarray
from six import iteritems as diter

def pandashandler(*args, **kwargs):
    """
    process a pysal model signature and convert an equation/formula pair into a
    pysal-specific object
    """
    if 'data' in kwargs:
        data = kwargs.pop('data')
        if 'formula' in kwargs: #check if it's provided as kwarg and catch
            formula_like = kwargs.pop('formula')
        elif isinstance(args[0], str): #if not, "standard" is as first arg
            formula_like = args[0]
        else:
            print('Formula not found, even though data provided.')
            print('Using legacy spreg behavior.')
            rargs = args[1:]
            rkwargs = kwargs

        if '||' in formula_like:
            mu, inst = formula_like.split('||')
            y, X = p.dmatrices(mu + '-1' , data=data)
            yend, q  = p.dmatrices(inst + '-1', data=data)
            rargs = [y,X,yend,q]
            rargs = [asarray(i) for i in rargs]
        else:
            y, X = p.dmatrices(formula_like + '-1', data=data)
            rargs = [asarray(y), asarray(X)]
            rkwargs = kwargs
    else:
        print('No data provided, avoiding formula parsing')
        rargs = args
        rkwargs = kwargs

    return rargs, rkwargs

#def model(eqn, *args, data=df, **kwargs)
class Model(object):
    """
    le model manager
    """
    def __init__(self, *args, **kwargs):
        mtype = kwargs.pop('mtype', sr.OLS)
        self._mtype = mtype
        if isinstance(mtype, str):
            mtype = sr.__dict__[mtype] 
        self._fit = kwargs.pop('fit', True)

        if isinstance(args[0], str):
            args, kwargs = pandashandler(*args, **kwargs)

        if self._fit:
            self._called = mtype(*args, **kwargs)
            for name in dir(self._called):
                self.__dict__.update({name:eval('self._called.{}'.format(name))})
