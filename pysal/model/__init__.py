_SUBMODULES = {"access", "mgwr", "spglm", "spint", "spreg", "tobler", "spopt"}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        module = importlib.import_module(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))

