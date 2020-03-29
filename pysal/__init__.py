__version__ = '2.3.0'

from .base import memberships, federation_hierarchy, versions
import importlib
import sys
import importlib
from types import ModuleType


class LazyLoader(ModuleType):
    @property
    def lib(self):
        if not self.__dict__.get('lib'):
            self.__dict__['lib'] = importlib.import_module('.lib', __package__)

        return self.__dict__['lib']

    @lib.setter
    def lib(self, mod):
        self.__dict__['examples'] = mod

    @property
    def explore(self):
        if not self.__dict__.get('explore'):
            self.__dict__['explore'] = importlib.import_module('.explore', __package__)

        return self.__dict__['explore']

    @explore.setter
    def explore(self, mod):
        self.__dict__['examples'] = mod


    @property
    def model(self):
        if not self.__dict__.get('model'):
            self.__dict__['model'] = importlib.import_module('.model', __package__)

        return self.__dict__['model']

    @model.setter
    def model(self, mod):
        self.__dict__['examples'] = mod


    @property
    def viz(self):
        if not self.__dict__.get('viz'):
            self.__dict__['viz'] = importlib.import_module('.viz', __package__)

        return self.__dict__['viz']

    @viz.setter
    def viz(self, mod):
        self.__dict__['examples'] = mod



old = sys.modules[__name__]
new = LazyLoader(__name__)
new.__path__ = old.__path__

for k, v in list(old.__dict__.items()):
    new.__dict__[k] = v

sys.modules[__name__] = new
