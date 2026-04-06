from pkgutil import iter_modules
from pysal import federation_hierarchy


def module_exists(module_name):
    return module_name in (name for loader, name, ispkg in iter_modules())

def test_imports():
    for layer in federation_hierarchy:
        packages = federation_hierarchy[layer]
        for package in packages:
            assert module_exists(package), f"{package} not installed." 

import pytest
from pysal.lib.common import simport

def test_simport_empty_string():
    with pytest.raises(Exception):
        simport("")

def test_simport_none():
    with pytest.raises(Exception):
        simport(None)
