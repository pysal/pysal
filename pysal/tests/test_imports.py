import pkgutil
import pysal


def test_imports():
    __all__ = []
    for loader, module_name, is_pkg in  pkgutil.walk_packages(pysal.__path__):
        __all__.append(module_name)
        _module = loader.find_module(module_name).load_module(module_name)
        globals()[module_name] = _module