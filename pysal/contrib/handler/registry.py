import pysal
from types import ModuleType, ClassType
from six import iteritems as diter

def _find_baseclasses():
    classes = dict()
    baseclasses = dict()
    submods = dict()
    for obname in dir(pysal.spreg):
        ob = eval('pysal.spreg.{}'.format(obname))
        if isinstance(ob, ModuleType):
            if ob.__package__.startswith("pysal"):
                submods.update({obname:ob})
        elif isinstance(ob, ClassType):
            classes.update({obname:ob})
            if ob.__name__.startswith('Base'):
                baseclasses.update({obname:ob})
    for modname, mod in diter(submods):
        candidates = dict()
        for clname in dir(mod):
            cl = mod.__dict__[clname]
            if isinstance(cl, ClassType):
                try:
                    if cl.__name__.startswith('Base'):
                        candidates.update({cl.__name__:cl})
                except:
                    pass
        baseclasses.update({k:v for k,v in diter(candidates) if cl not in baseclasses})
    userclasses = {k:v for k,v in diter(classes) if k not in baseclasses and
                   'diagnostics' not in v.__module__}
    return baseclasses, userclasses

base, user = _find_baseclasses()

__all__ = dict()
__all__.update(base)
__all__.update(user)
#regimes = {cls for cls in user if 'regimes' in cls.__module__}

#if we go with something like a "base" and "user" submodule setup,
#it'd be as simple as flattening the subclasses out into those submodules. 
#for name, cl in diter(_find_baseclasses()[0]):
#    exec('{n} = {cls}'.format(n=name, cls=cl))
