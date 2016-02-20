import copy
import registry as sr
from pysal.weights import W
try:
    import patsy as p
except:
    p = None
from numpy import array, ndarray, asarray
from six import iteritems as diter

#would like to just wrap this in the opt decorator...
def pandashandler(formula_like, data):
    """
    process a pysal model signature and convert an equation/formula pair into a
    pysal-specific object
    """
    if '||' in formula_like:
        mu, inst = formula_like.split('||')
        y, X = p.dmatrices(mu + '-1' , data=data)
        yend, q  = p.dmatrices(inst + '-1', data=data)
        rargs = [y,X,yend,q]
        rargs = [asarray(i) for i in rargs]
    else:
        y, X = p.dmatrices(formula_like + '-1', data=data)
        rargs = [asarray(y), asarray(X)]

    return rargs

#def model(eqn, *args, data=df, **kwargs)
class Model(object):
    """
    The model manager
    
    arguments that sit above the pysal model API:

    mtype : string mapping to the function called from spreg
    fit : Bool denoting whether to actually apply the mtype to the data provided
          immediately.

    an example call would look like:

        >>> Model(y,X,W, mtype='ML_Lag')
        >>> Model(y,X,W, mytpe='OLS_Regimes')
    """
    def __init__(self, *args, **kwargs):
        self._cache = {}
        mtype = kwargs.pop('mtype', 'OLS')
        self._mtype = mtype
        self._mfunc = sr._everything[mtype] 
        self._fit = kwargs.pop('fit', True)

        if isinstance(args[0], str):
            formula = args[0]
            data = kwargs.pop('data')
            matrices = pandashandler(formula, data)
        elif 'formula' in kwargs.keys() and 'data' in kwargs.keys():
            formula = kwargs.pop('formula')
            data = kwargs.pop('data')
            matrices = pandashandler(formula, data)
        else:
            matrices = [arg for arg in args if not isinstance(arg, W)]
        
        args = matrices + [arg for arg in args if isinstance(arg, W)]

        if self._fit:
            self._called = self._mfunc(*args, **kwargs)
            for name in dir(self._called):
                try:
                    exec('self.{n} = self._called.{n}'.format(n=name))
                except:
                    print("Assigning {a} from {s} to {d} failed!".format(a=name,
                                                                             s=self._called,
                                                                         d=self))
    
#need to still pass names down from formula into relevant pysal arguments

if __name__ == '__main__':
    import pysal as ps

    dbf = ps.open(ps.examples.get_path('columbus.dbf'))
    y, X = dbf.by_col_array(['HOVAL']), dbf.by_col_array(['INC', 'CRIME'])
    Wcol = ps.open(ps.examples.get_path('columbus.gal')).read()
    mod1 = sr.OLS(y,X)
    hmod1 = Model(y,X)

    mod2 = sr.OLS(y,X,Wcol)
    hmod2 = Model(y,X,Wcol)

    mod3 = sr.ML_Lag(y,X,Wcol)
    hmod3 = Model(y,X,Wcol, mtype='ML_Lag')

    mod4 = sr.ML_Error(y,X,Wcol)
    hmod4 = Model(y,X,Wcol,mtype='ML_Error')

    #real power comes from this, though
#    import geopandas as gpd
#    
#    df = gpd.read_file(ps.examples.get_path('columbus.dbf'))
#
#    hmod1_pd = Model('HOVAL ~ INC + CRIME', data=data)
#    mod5 = sr.TSLS('HOVAL ~ INC + CRIME')
