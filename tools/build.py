"""
Build requirements file for meta package from latest stable subpackages on pypi

"""
from yolk.pypi import CheeseShop
packages = """
libpysal
esda
giddy
inequality
pointpats
segregation
spaghetti
mgwr
spglm
spint
spreg
spvcm
tobler
mapclassify
splot
"""

other_reqs = """
urllib3<1.25
python-dateutil<=2.8.0
pytest
pytest-cov
coverage
"""

def _get_latest_version_number(package_name):
    pkg, all_versions = CheeseShop().query_versions_pypi(package_name)
    if len(all_versions):
        return all_versions[0]
    return None

def build_requirements():
    """
    Write out requirements.txt file with pinning information
    """
    lines = []
    for package in packages.split():
        version = _get_latest_version_number(package)
        print(package, version )
        if package != 'spvcm':
            lines.append(f'{package}>={version}')
        else:
            lines.append(f'{package}=={version}')

    for package in other_reqs.split():
        lines.append(package)

    with open('requirements.txt', 'w') as req:
        req.write("\n".join(lines))
