import copy
from registry import user as mr, clstypes 
from pysal.weights import W
try:
    import patsy as p
except:
    p = None
from numpy import array, ndarray, asarray
from pysal.common import iteritems as diter
import inspect

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
        name_y, name_x = mu.strip(' ').split('~')
        name_x = name_x.split('+')
        name_yend, name_q = inst.strip(' ').split('~')
        name_yend = [name_yend]
        name_q = name_q.split('+')
        names = {"name_y":name_y,
                 "name_x":name_x, 
                 "name_yend":name_yend,
                 "name_q":name_q}
    else:
        y, X = p.dmatrices(formula_like + '-1', data=data)
        rargs = [asarray(y), asarray(X)]
        name_y, name_x = formula_like.strip(' ').split('~')
        name_x = name_x.split('+')
        names = {"name_y":name_y,
                 "name_x":name_x}

    return rargs, names

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
        self._outers = {}
        mtype = kwargs.pop('mtype', 'OLS')
        if mtype.startswith('Base'):
            raise Exception('Only Userclasses can be fit using the handler')
        mtype = mtype
        mfunc = mr.__dict__[mtype]
        fit = kwargs.pop('fit', True)
        ins = inspect.getargspec(mfunc.__init__)
        req = len(ins.args) - (len(ins.defaults)+1 )

        if isinstance(args[0], str):
            if len(args) > 1:
                raise TypeError('When a formula is used as the first argument,'
                                ' all other arguments must be named. Consult {}'
                                ' for argument names'.format(self._mfunc.__name__))
            formula = args[0]
            data = kwargs.pop('data')
            matrices, names = pandashandler(formula, data)
            kwargs.update(names)
        elif 'formula' in kwargs.keys() and 'data' in kwargs.keys():
            formula = kwargs.pop('formula')
            data = kwargs.pop('data')
            matrices, names = pandashandler(formula, data)
            kwargs.update(names)
        else:
            matrices = list(args[0:req])

        if fit:
            self._called = mfunc(*matrices, **kwargs)
            self._fit = True
        else:
            self._outers['mtype'] = mtype
            self._outers['mfunc'] = mfunc
            self._fit = False
    
    @property
    def __dict__(self):
        inners = self._called.__dict__
        obligations = [x for x in dir(self._mfunc) if not x.startswith('_')]
        obligations = {k:self._called.__getattribute__(k) for k in obligations}
        outers = self._outers
        alldict = dict()
        alldict.update(inners)
        alldict.update(obligations)
        alldict.update(outers)
        return alldict
    
    @__dict__.setter
    def __dict__(self, key, val):
        self._outers[key] = val

    @property
    def _mfunc(self):
        return type(self._called)
    
    @property
    def _mtype(self):
        return type(self._called).__name__

    def __getattr__(self, val):
        return self._called.__getattribute__(val)

if __name__ == '__main__':
    import pysal as ps

    dbf = ps.open(ps.examples.get_path('columbus.dbf'))
    y, X = dbf.by_col_array(['HOVAL']), dbf.by_col_array(['INC', 'CRIME'])
    Wcol = ps.open(ps.examples.get_path('columbus.gal')).read()
    mod1 = mr.OLS(y,X)
    hmod1 = Model(y,X)

    mod2 = mr.OLS(y,X,w=Wcol, spat_diag=True, moran=True)
    hmod2 = Model(y,X,w=Wcol, spat_diag=True, moran=True)

    mod3 = mr.ML_Lag(y,X,Wcol)
    hmod3 = Model(y,X,Wcol, mtype='ML_Lag')

    mod4 = mr.ML_Error(y,X,Wcol)
    hmod4 = Model(y,X,w=Wcol,mtype='ML_Error')

    #real power comes from this, though
#    import geopandas as gpd
#    
#    df = gpd.read_file(ps.examples.get_path('columbus.dbf'))
#
#    hmod1_pd = Model('HOVAL ~ INC + CRIME', data=data)
#    mod5 = sr.TSLS('HOVAL ~ INC + CRIME')
