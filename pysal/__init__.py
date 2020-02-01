__version__='v2.2.0rc'
from . import lib
from . import explore
from . import viz
from . import model


def check_versions():
    from .base import frozen_packages
    import warnings
    frozen_packages.pop('spvcm')
    current_packages = {}
    for pkg in frozen_packages.keys():
        exec(f'import {pkg}')
        current_packages[pkg] = eval(f'{pkg}.__version__')
    diffs = 0
    for pkg in frozen_packages.keys():
        if frozen_packages[pkg] != current_packages[pkg]:
            diffs += 1
            msg = (
                f"The last meta package for PySAL included {pkg} "
                f"at version {frozen_packages[pkg]}, but you have "
                f"version {current_packages[pkg]}."
                )
            warnings.warn(msg)
    if diffs == 0:
        print(f'PySAL subpackages all pinned to meta release {__version__}.')
