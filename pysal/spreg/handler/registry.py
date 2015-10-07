import pysal
from types import ModuleType, ClassType
from six import iteritems as diter

def _find_bcs():
    classes = dict()
    bcs = dict()
    ucs = dict()
    submods = dict()
    for obname in dir(pysal.spreg):
        ob = pysal.spreg.__dict__[obname]
        if isinstance(ob, ModuleType):
            if ob.__package__.startswith("pysal"):
                submods.update({obname:ob})
        elif isinstance(ob, ClassType):
            classes.update({obname:ob})
            if ob.__name__.startswith('Base'):
                bcs.update({obname:ob})
    for modname, mod in diter(submods):
        basecands = dict()
        for clname in dir(mod):
            cl = mod.__dict__[clname]
            if isinstance(cl, ClassType):
                try:
                    if cl.__name__.startswith('Base'):
                        if cl not in bcs:
                            bcs.update({cl.__name__:cl})
                        else:
                            classes.update({cl.__name__:cl})
                except:
                    pass
    ucs.update({k:v for k,v in diter(classes) if (
                any([issubclass(v, bc) for bc in bcs.values()])
                and (k not in bcs))
                or k.endswith('Regimes')})
    return bcs, ucs

base, user = _find_bcs()
_everything = base.copy()
_everything.update(user)

for k,v in diter(base):
    exec('{k} = {v}'.format(k=k,v=v))
for k,v in diter(user):
    exec('{k} = {v}'.format(k=k,v=v))

__all__ = list()
__all__.extend(base.keys())
__all__.extend(user.keys())
#regimes = {cls for cls in user if 'regimes' in cls.__module__}

#if we go with something like a "base" and "user" submodule setup,
#it'd be as simple as flattening the subclasses out into those submodules. 
#for name, cl in diter(_find_bcs()[0]):
#    exec('{n} = {cls}'.format(n=name, cls=cl))
