"""
Base information for pysal meta package
"""

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property

federation_hierarchy = {
    "explore": [
        "esda",
        "giddy",
        "segregation",
        "pointpats",
        "inequality",
        "spaghetti",
        "access",
        "momepy",
    ],
    "model": ["spreg", "spglm", "tobler", "spint", "mgwr", "access", "spopt"],
    "viz": ["splot", "mapclassify"],
    "lib": ["libpysal"],
}

memberships = {}
for key in federation_hierarchy:
    for package in federation_hierarchy[key]:
        memberships[package] = key


def _installed_version(package):
    """Get the installed version of a package.

    Parameters
    ----------
    package : str
        Name of the package to check.

    Returns
    -------
    str
        Version string if available, 'NA' otherwise.
    """
    try:
        mod = importlib.import_module(package)
        return getattr(mod, "__version__", "NA")
    except (ModuleNotFoundError, ImportError):
        return "NA"


def _installed_versions():
    packages = list(memberships.keys())
    max_workers = min(len(packages), 8)

    ver = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_package = {
                executor.submit(_installed_version, pkg): pkg
                for pkg in packages
            }

            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    ver[package] = future.result(timeout=5.0)
                except Exception:
                    ver[package] = "NA"
    except (RuntimeError, OSError):
        for package in packages:
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
        s = f"{package:>12} | {installed:>15} | {released:>15} | {match:>5}"
        table.append(s)
        table.append("-" * len(s))
        for package in self.installed:
            installed = self.installed[package]
            released = self.released[package]
            match = installed == released
            s = f"{package:>12} | {installed:>15} | {released:>15} | {match:>5}"
            table.append(s)
        print("\n".join(table))


versions = Versions()
