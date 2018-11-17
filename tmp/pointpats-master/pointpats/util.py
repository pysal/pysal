__all__ = ['cached_property']
import functools


def cached_property(fun):
    """A memoize decorator for class properties.

    Adapted from: http://code.activestate.com/recipes/576563-cached-property/
    """
    @functools.wraps(fun)
    def get(self):
        try:
            return self._cache[fun]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        ret = self._cache[fun] = fun(self)
        return ret
    return property(get)
