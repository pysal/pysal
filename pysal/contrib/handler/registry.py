import pysal
import types as ty
import sys

PY3 = sys.version_info.major > 2
from pysal.common import iteritems as diter

if not PY3:
    clstypes = (type, ty.ClassType)
else:
    clstypes = type

def _find_bcs():
    classes = dict()
    bcs = dict()
    ucs = dict()
    submods = dict()
    for obname, ob in diter(pysal.spreg.__dict__):
        if isinstance(ob, ty.ModuleType):
            if ob.__package__.startswith("pysal"):
                submods.update({obname:ob})
        elif isinstance(ob, clstypes):
            classes.update({obname:ob})
            if ob.__name__.startswith('Base'):
                bcs.update({obname:ob})
    for modname, mod in diter(submods):
        basecands = dict()
        for clname in dir(mod):
            cl = mod.__dict__[clname]
            if isinstance(cl, clstypes):
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

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

base, user = _find_bcs()
base = Namespace(**base)
user = Namespace(**user)
everything = Namespace(**base.__dict__)
everything.__dict__.update(user.__dict__)

#if we go with something like a "base" and "user" submodule setup,
#it'd be as simple as flattening the subclasses out into those submodules. 
#for name, cl in diter(_find_bcs()[0]):
#    exec('{n} = {cls}'.format(n=name, cls=cl))
