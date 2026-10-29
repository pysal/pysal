import importlib
import pkgutil

import pytest


def _wrapper_modules():
    for layer in ("explore", "model", "viz"):
        pkg = importlib.import_module(f"pysal.{layer}")
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.ispkg:
                yield f"pysal.{layer}.{info.name}"


@pytest.mark.parametrize("module", list(_wrapper_modules()))
def test_wrapper_imports(module):
    importlib.import_module(module)
