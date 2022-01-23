"""
Base information for pysal meta package
"""


federation_hierarchy = {
    'explore': ['esda', 'giddy', 'segregation',
                'pointpats', 'inequality',
                 'spaghetti', 'access', 'momepy'],
    'model': ['spreg', 'spglm', 'tobler', 'spint',
              'mgwr', 'spvcm', 'access', 'spopt'],
    'viz': ['splot', 'mapclassify'],
    'lib': ['libpysal']
}

memberships = {}
for key in federation_hierarchy:
    for package in federation_hierarchy[key]:
        memberships[package] = key



class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

def _installed_version(package):
    try:
        exec(f'import {package}')
    except ModuleNotFoundError:
        v = 'NA'
    try:
        v = eval(f'{package}.__version__')
    except AttributeError:
        v = 'NA'
    return v

def _installed_versions():
    ver = {}
    for package in memberships.keys():
        ver[package] = _installed_version(package)
    return ver

def _released_versions():
    from .frozen import frozen_packages
    return frozen_packages


class Versions:
    @cached_property
    def installed(self):
        """
        Inventory versions of pysal packages that are installed

        Attributes
        ----------
        installed: dict
                   key is package name, value is version string
        """
        return _installed_versions()

    @cached_property
    def released(self):
        """
        Inventory versions of pysal packages that are released in the meta
        package.

        Attributes
        ----------
        released: dict
                   key is package name, value is version string
        """

        return _released_versions()

    def check(self):
        """
        Print a tabular string that reports installed and released versions of
        PySAL packages.
        """
        table = []
        package = "Package"
        installed = "Installed"
        released = "Released"
        match = "Match"
        s = f'{package:>12} | {installed:>15} | {released:>15} | {match:>5}'
        table.append(s)
        table.append("-"*len(s))
        for package in self.installed:
            installed = self.installed[package]
            released = self.released[package]
            match = installed == released
            s = f'{package:>12} | {installed:>15} | {released:>15} | {match:>5}'
            table.append(s)
        print("\n".join(table))


versions = Versions()

