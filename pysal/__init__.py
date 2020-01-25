__version__='v2.2.0rc'
from . import lib
from . import explore
from . import viz
from . import model


def check_versions():
    from .base import frozen_packages
    import warnings
    current_packages = {}
    for pkg in frozen_packages.keys():
        exec(f'import {pkg}')
        current_packages[pkg] = eval(f'{pkg}.__version__')
    for pkg in frozen_packages.keys():
        if frozen_packages[pkg] != current_packages[pkg]:
            msg = (
                f"The last meta package for PySAL included {pkg} "
                f"at version {frozen_packages[pkg]}, but you have "
                f"a more recent version {current_packages[pkg]}."
                )
            warnings.warn(msg)
