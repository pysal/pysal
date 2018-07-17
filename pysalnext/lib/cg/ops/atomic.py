from . import _accessors as _a
from . import _shapely as _s

# prefer access to shapely computation
_all = dict()
_all.update(_s.__dict__)
_all.update(_a.__dict__)

globals().update({_k:_v for _k,_v in list(_all.items()) if not _k.startswith('_')})
_preferred = _a
