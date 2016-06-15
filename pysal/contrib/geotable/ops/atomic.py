import _accessors as _a
import _shapely as _s

# prefer access to shapely computation
_s.__dict__.update(_a.__dict__)

globals().update(_s.__dict__)
